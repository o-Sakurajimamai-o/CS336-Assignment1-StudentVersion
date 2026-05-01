import os
import time
import json
import regex as re
from tqdm import tqdm
import multiprocessing
from typing import BinaryIO
from collections import Counter
from multiprocessing import Pool

"""Training utilities for a byte-level BPE tokenizer."""


HEX_MERGES_HEADER = "# format: hex-v1"


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pretokenization_wrapper(args):
    """Unpack multiprocessing arguments for pretokenization."""
    return pretokenization(*args)

def pretokenization(file, start, end, special_tokens):
    """Count pre-tokenized byte tuples in one file chunk."""
    with open(file, "rb") as f:
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token

    # Attention, Windows will automaticly add '\r' to your seq
    chunk = chunk.replace("\r", "")

    # get my pid name
    process_name = multiprocessing.current_process().name
    # print(f'Process {process_name} is working')

    # make special token to a list which can't be merge
    escaped_token = [re.escape(tok) for tok in special_tokens]
    special_tokens = "|".join(escaped_token) if escaped_token else None

    # print(f'special token is {special_tokens}')

    if special_tokens is not None:
        chunks = re.split(special_tokens, chunk)
    else:
        chunks = [chunk]    
    # print(f'chunk size is {len(chunks)}')

    # count char in a word
    tot = Counter()
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for chunk in chunks:
        # split a seq, eg("I am a handsome boy" -> "I", " am", " a", " handsome", " boy")
        for match in re.finditer(PAT, chunk):
            # get a word
            word = match.group()
            # word -> int
            word = word.encode("utf-8")

            tup = tuple(bytes([b]) for b in word)
            tot[tup] += 1
    
    return tot


def train_bpe(file, vocab_size, special_tokens):
    """Train a byte-level BPE vocabulary and return (vocab, merges)."""

    starttime = time.time()

    # num_workers processes to do the pretokenization
    num_workers = max(1, multiprocessing.cpu_count() - 1)

    file_size = os.path.getsize(file)
    num_chunks= max(1, file_size // (150 * 1024 * 1024))

    file_size = os.path.getsize(file)
    
    with open(file, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")

    print(f'{num_workers} processes working on {len(boundaries)-1} chunks of {file}')
    tasks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        tasks.append((file, start, end, special_tokens))

    total = Counter()
    with Pool(processes=num_workers) as pool:
        for tot in tqdm(pool.imap_unordered(pretokenization_wrapper, tasks), total=len(tasks), desc="Phase 1: Pretokenization"):
            total.update(tot)

    print("pretokenization done")

    print(f'seq length: {len(total)}')
    # for word, count in total.items():
    #     print(f'{word}: {count}')

    # Start with the 256 raw byte tokens.
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    
    # Add special tokens after the base byte vocabulary.
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
   

#============================================================================================================
    # Count adjacent byte-pair frequencies and build an inverted index from
    # each pair to the words containing that pair.
    count = {}
    pair_to_word = {}
    for word, cnt in total.items():
        for i in range(len(word) - 1):
            tmp_pair = (word[i], word[i + 1])
            count[tmp_pair] = count.get(tmp_pair, 0) + cnt

            # create inverted index
            if tmp_pair not in pair_to_word:
                pair_to_word[tmp_pair] = set()
            pair_to_word[tmp_pair].add(word)
    target_merges = vocab_size - len(vocab)
#============================================================================================================
    
    pbar = tqdm(total=target_merges, desc="Phase 2: BPE Merging")
    merges = []
    # merge operation
    while len(vocab) < vocab_size:
        if not count:
            break
        # print(count)
        if len(count) > 0:
            pair = max(count.items(), key=lambda x : (x[1], x[0]))[0]
        else:
            pair = None
            break
        
        merges.append(pair)
        vocab[len(vocab)] = pair[0] + pair[1]

        # Only words containing the selected pair need to be updated.
        update = list(pair_to_word.get(pair, set()))
        for word in update:
            cnt = total[word]

            # if this seq have best-pair probably, then sub every pair first, and add new-pair
            for i in range(len(word) - 1):
                old_pair = (word[i], word[i + 1])
                if old_pair in count:
                    count[old_pair] -= cnt
                    if count[old_pair] <= 0:
                        del count[old_pair]

                if old_pair in pair_to_word and word in pair_to_word[old_pair]:
                    pair_to_word[old_pair].remove(word)
                    if not pair_to_word[old_pair]:
                        del pair_to_word[old_pair]

            # merge, if tmp_pair = pair
            tuplist = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                    tuplist.append(pair[0] + pair[1])
                    i += 2
                else:
                    tuplist.append(word[i])
                    i += 1

            # count the frequency of every tuple(merged), this seq's number is cnt           
            tup = tuple(tuplist)
            del total[word]
            total[tup] = total.get(tup, 0) + cnt

            # add new-pair
            for i in range(len(tup) - 1):
                new_pair = (tup[i], tup[i + 1])
                count[new_pair] = count.get(new_pair, 0) + cnt

                if new_pair not in pair_to_word:
                    pair_to_word[new_pair] = set()
                pair_to_word[new_pair].add(tup)

        pbar.update(1)

    pbar.close()
    endtime = time.time()
    
    cost = endtime - starttime
    m, s = divmod(cost, 60)
    h, m = divmod(m, 60)

    print(f'Done, cost : {h} hours and {m} minutes and {s} sceonds')

    longest_token = max(vocab.values(), key=len)
    print(f'the longest token is {longest_token}, size: {len(longest_token)}')
    # print(f'Vocab : {vocab}')
    # print(f'merges : {merges}')
    return vocab, merges


def save_vocab(vocab: dict[int, bytes], vocab_path: str) -> None:
    """Save vocab bytes losslessly as hex strings."""
    payload = {
        "format": "hex-v1",
        "vocab": {str(k): token.hex() for k, token in vocab.items()},
    }
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_merges(merges: list[tuple[bytes, bytes]], merges_path: str) -> None:
    """Save merge pairs losslessly as tab-separated hex strings."""
    with open(merges_path, "w", encoding="utf-8") as f:
        f.write(f"{HEX_MERGES_HEADER}\n")
        for left, right in merges:
            f.write(f"{left.hex()}\t{right.hex()}\n")


if __name__ == '__main__':
    file = r"D:\DeepLearningProject\CS336\assignment1-basics\data\TinyStoriesV2-GPT4-train.txt"
    special_tokens = []
    special_tokens.append("<|endoftext|>")
    vocab, merges = train_bpe(file, 10000, special_tokens)

    # 1. save Vocab
    save_vocab(vocab, "owt_vocab.json")

    # 2. save merges to txt
    save_merges(merges, "owt_merges.txt")

    print("Save Done")
