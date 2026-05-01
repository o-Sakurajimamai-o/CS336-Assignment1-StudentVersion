import os
import torch
import argparse
from torch import nn

from cs336_basics.Chapter2.Tokenizer import Tokenizer
from cs336_basics.Chapter3.TransformerLM import TransformerLM
from cs336_basics.Chapter5.checkpointing import load_checkpoint
from cs336_basics.Chapter6.decoding import decode

parser = argparse.ArgumentParser(description="Train a Transformer LM")
# ===================================================================================================

# record
parser.add_argument('--use_wandb', action='store_true', help="wether use WandB")
parser.add_argument('--wandb_project', type=str, default="Tansformer LM")
parser.add_argument('--wandb_name', type=str, default=None, help="Transformer LM")

# the args of structed model 
parser.add_argument('--vocab_size', type=int, default=10001)
parser.add_argument('--context_length', type=int, default=256)
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--num_heads', type=int, default=16)
parser.add_argument('--d_ff', type=int, default=1344)

# training hyperparameters
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--max_iters', type=int, default=50000)
parser.add_argument('--lr_max', type=float, default=3e-4)
parser.add_argument('--lr_min', type=float, default=3e-5)
parser.add_argument('--warmup_iters', type=int, default=1500)
parser.add_argument('--grad_clip', type=float, default=1.0)

# file path and log
parser.add_argument('--train_data', type=str, default=r"D:\DeepLearningProject\CS336\assignment1-basics\data\train.bin")
parser.add_argument('--val_data', type=str, default=r"D:\DeepLearningProject\CS336\assignment1-basics\data\val.bin")
parser.add_argument('--out_dir', type=str, default='./checkpoints')
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--save_interval', type=int, default=5000)
parser.add_argument('--resume', type=str, default=None)

# validation
parser.add_argument('--eval_interval', type=int, default=500)
parser.add_argument('--eval_iters', type=int, default=20)

args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Predicting on {device}')

# load model
model = TransformerLM(
    args.vocab_size, args.context_length,
    args.d_model, args.num_layers, args.num_heads,
    args.d_ff
).to(device)

model_path = r"D:\DeepLearningProject\CS336\assignment1-basics\cs336_basics\Chapter5\checkpoints\checkpoint_step_49999.pt"
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

vocab_filepath = r"D:\DeepLearningProject\CS336\assignment1-basics\cs336_basics\tinystories_vocab.json"
merges_filepath = r"D:\DeepLearningProject\CS336\assignment1-basics\cs336_basics\tinystories_merges.txt"
special_tokens = []
special_tokens.append("<|endoftext|>")
tokenizer = Tokenizer.from_files(
    vocab_filepath=vocab_filepath,
    merges_filepath=merges_filepath,
    special_tokens=special_tokens
)

print("Give me a little seq, and I will give you a whole story!")
prompt = input()
prompt = tokenizer.encode(prompt)
prompt = torch.tensor(prompt, dtype=torch.long).unsqueeze(0).to(device)

with torch.no_grad():
    answer = decode(model, prompt)

answer = answer[0].tolist()
answer = tokenizer.decode(answer)

print(answer)
