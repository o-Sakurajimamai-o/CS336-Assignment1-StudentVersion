import json
import regex as re
from typing import Iterable, Iterator

class Tokenizer:
    """Byte-pair encoding tokenizer compatible with the CS336 assignment tests."""

    def __init__(self, vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.byte_to_int = {v: k for k, v in vocab.items()}
        # Lower rank means the pair was learned earlier and should be applied first.
        self.merge_rank = {pair: i for i, pair in enumerate(self.merges)}


    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """Load a tokenizer from a JSON vocabulary and a text merges file."""
        
        # we need to load vocab.json to a dict of py
        vocab = {}
        with open(vocab_filepath, "rb") as f:
            row = json.load(f)

            for k, v in row.items():
                vocab[int(k)] = v.encode("utf-8")

        # then we do the same to merges, but a list
        merges = []
        with open(merges_filepath, "rb") as f:
            for line in f:
                line = line.rstrip(b'\r\n')
                pair = line.split(b" ")

                if len(pair) == 2:
                    merges.append((pair[0], pair[1]))
        
        # print(f'vocab type: {type(vocab)}')
        # print(f'merges type: {type(merges)}')

        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        """Convert text into token ids, keeping configured special tokens intact."""
        
        special_tokens_list = self.special_tokens or []
        special_tokens_list = sorted(special_tokens_list, key=len, reverse=True)
        escaped_token = [re.escape(tok) for tok in special_tokens_list]
        special_tokens_list = f"({'|'.join(escaped_token)})" if escaped_token else None

        if special_tokens_list is not None:
            chunks = re.split(special_tokens_list, text)
        else:
            chunks = [text]

        # chunk: [seq1 special seq2] -> [seq1, sepcial, seq2]
        # print(chunks)    

        IDs = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for chunk in chunks:
            if chunk is None: continue
            if self.special_tokens and chunk in self.special_tokens:
                IDs.append(self.byte_to_int[chunk.encode('utf-8')])
                continue

            # split a seq, eg("I am a handsome boy" -> "I", " am", " a", " handsome", " boy")
            for match in re.finditer(PAT, chunk):
                # get a word
                word = match.group()
                # get bytes
                word = word.encode('utf-8')
                # first, we make word to a list of bytes
                b_list = [bytes([b]) for b in word]

                # Repeatedly merge the available pair with the earliest learned rank.
                while True:
                    tmp_list = []
                    index = float('inf')
                    pair = None
                    for i in range(len(b_list) - 1):
                        b = (b_list[i], b_list[i + 1])
                        # here, we will find the most frequent pair, because of hash, so we have O(1) to find rank
                        if b in self.merge_rank and index > self.merge_rank[b]:
                            index = self.merge_rank[b]
                            pair = b

                    # then we merge any pair which is the bestpair
                    if pair is None: break
                    i = 0
                    while i < len(b_list):
                        if i + 1 < len(b_list) and (b_list[i], b_list[i + 1]) == pair: 
                            tmp_list.append(b_list[i] + b_list[i + 1])
                            i += 2
                        else: 
                            tmp_list.append(b_list[i])
                            i += 1
                    b_list = tmp_list
                # print(b_list)
                for b in b_list:
                    IDs.append(self.byte_to_int[b])


        # print(IDs)

        return IDs
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """Stream token ids from an iterable of text chunks without joining them."""
        for line in iterable:
            ids = self.encode(line)

            for token in ids:
                yield token

    def decode(self, ids: list[int]) -> str:
        """Convert token ids back into a UTF-8 string."""
        tokens = [self.vocab[token] for token in ids]
        seq = b"".join(tokens)
        # print(seq)
        return seq.decode('utf-8', errors="replace")


if __name__ == '__main__':
    vocab_filepath = r"D:\DeepLearningProject\CS336\assignment1-basics\cs336_basics\tinystories_vocab.json"
    merges_filepath = r"D:\DeepLearningProject\CS336\assignment1-basics\cs336_basics\tinystories_merges.txt"
    special_tokens = []
    special_tokens.append("<|endoftext|>")
    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_filepath,
        merges_filepath=merges_filepath,
        special_tokens=special_tokens
    )

    text = "I am a good boy, and I konw everything!" \
    "<|endoftext|>" \
    "I am not a good boy"
    encode = tokenizer.encode(text)
    decode = tokenizer.decode(encode)
