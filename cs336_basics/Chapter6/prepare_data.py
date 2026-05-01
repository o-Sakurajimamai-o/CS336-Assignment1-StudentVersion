import numpy as np

# 1. 导入你刚刚写的 Tokenizer 类
from cs336_basics.Chapter2.Tokenizer import Tokenizer

TEXT_CHARS_PER_BATCH = 8 * 1024 * 1024
TOKENS_PER_WRITE = 1_000_000
PROGRESS_EVERY = 10


def iter_text_batches(file_obj, target_chars=TEXT_CHARS_PER_BATCH):
    buffer = []
    buffered_chars = 0

    for line in file_obj:
        buffer.append(line)
        buffered_chars += len(line)

        if buffered_chars >= target_chars:
            yield "".join(buffer)
            buffer.clear()
            buffered_chars = 0

    if buffer:
        yield "".join(buffer)


def flush_token_buffer(token_buffer, out_file):
    if not token_buffer:
        return 0

    np.asarray(token_buffer, dtype=np.uint16).tofile(out_file)
    flushed = len(token_buffer)
    token_buffer.clear()
    return flushed


def encode_and_save(text_path, bin_path, tokenizer):
    print(f"Encoding {text_path} ... (streaming mode)")

    total_tokens = 0
    token_buffer = []

    with open(text_path, "r", encoding="utf-8") as text_file, open(bin_path, "wb") as bin_file:
        for batch_idx, text_batch in enumerate(iter_text_batches(text_file), start=1):
            token_buffer.extend(tokenizer.encode(text_batch))

            if len(token_buffer) >= TOKENS_PER_WRITE:
                total_tokens += flush_token_buffer(token_buffer, bin_file)

            if batch_idx % PROGRESS_EVERY == 0:
                print(f"Processed {batch_idx} text batches, wrote {total_tokens + len(token_buffer)} tokens so far ...")

        total_tokens += flush_token_buffer(token_buffer, bin_file)

    print(f"Total tokens: {total_tokens}")
    print(f"Saved to {bin_path} successfully!\n")

if __name__ == '__main__':
    # 2. 使用你的 from_files 方法加载刚才训练好的密码本
    # 注意路径要写对
    tokenizer = Tokenizer.from_files(
        vocab_filepath=r"D:\DeepLearningProject\CS336\assignment1-basics\cs336_basics\Chapter2\tinystories_vocab.json",
        merges_filepath=r"D:\DeepLearningProject\CS336\assignment1-basics\cs336_basics\Chapter2\tinystories_merges.txt",
        special_tokens=["<|endoftext|>"]
    )

    # 3. 翻译 Train 集
    train_txt = r"D:\DeepLearningProject\CS336\assignment1-basics\data\TinyStoriesV2-GPT4-train.txt"
    train_bin = r"D:\DeepLearningProject\CS336\assignment1-basics\data\train.bin"
    encode_and_save(train_txt, train_bin, tokenizer)

    # 4. 翻译 Valid 集
    valid_txt = r"D:\DeepLearningProject\CS336\assignment1-basics\data\TinyStoriesV2-GPT4-valid.txt"
    valid_bin = r"D:\DeepLearningProject\CS336\assignment1-basics\data\val.bin"
    encode_and_save(valid_txt, valid_bin, tokenizer)
