import os
import time
import torch
import wandb
import argparse
import numpy as np
from tqdm import tqdm

from cs336_basics.Chapter3.TransformerLM import TransformerLM
from cs336_basics.Chapter4.AdamW import AdamW
from cs336_basics.Chapter4.cross_entropy import cropss_entropy
from cs336_basics.Chapter4.gradient_clip import Gradient_cliping
from cs336_basics.Chapter4.learning_rate_schedule import cos_learning_rate_schedule_with_warmup
from cs336_basics.Chapter5.checkpointing import save_checkpoint, load_checkpoint
from cs336_basics.Chapter5.data_loading import data_loading


@torch.no_grad()
def validation(model, data, batch_size, context_length, device, eval_iters=20):

    model.eval() 
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = data_loading(data, batch_size, context_length, device)
        logits = model(x)
        loss = cropss_entropy(logits, y)
        losses[k] = loss.item()
    model.train() 

    return losses.mean().item()

def train():
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

# ===================================================================================================

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args)
        )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training on {device}')

    train_data = np.memmap(args.train_data, dtype=np.uint16, mode='r')
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode='r')

# ====================================================================================================  
    
    # initial model
    model = TransformerLM(
        args.vocab_size, args.context_length,
        args.d_model, args.num_layers, args.num_heads,
        args.d_ff
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr_max)

    if args.resume is not None:
        init_step = load_checkpoint(args.resume, model, optimizer)
        print(f'loading data from {args.resume}, iteration = {init_step}')
    else:
        init_step = 0
        print(f'initialize the model and optimizer weights')

# ====================================================================================================  
    
    starttime = time.time()
    pbar = tqdm(
        range(init_step, args.max_iters), 
        initial=init_step, 
        total=args.max_iters, 
        desc="Training",
        dynamic_ncols=True
    )

    tokens_per_step = args.batch_size * args.context_length
    steps_per_epoch = max(1, len(train_data) // tokens_per_step) 
    total_epochs = args.max_iters // steps_per_epoch

    for step in pbar:
        
        current_epoch = (step // steps_per_epoch) + 1
        pbar.set_description(f"Epoch [{current_epoch}/{total_epochs}]")
        
        model.train()
        lr = cos_learning_rate_schedule_with_warmup(
            step, args.lr_max, args.lr_min, 
            args.warmup_iters, args.max_iters
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if step % args.eval_interval == 0:
            val_loss = validation(model, val_data, args.batch_size, args.context_length, device, args.eval_iters)
            train_loss = validation(model, train_data, args.batch_size, args.context_length, device, args.eval_iters)
            
            elapsed_time = time.time() - starttime
            pbar.write(f"Step {step:4d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if args.use_wandb:
                wandb.log({
                    "step": step,
                    "wall_clock_time_seconds": elapsed_time,
                    "loss/train_eval": train_loss,
                    "loss/val": val_loss,
                    "lr": lr
                }, step=step) 

        x, y = data_loading(
            train_data, args.batch_size,
            args.context_length, device=device
        )
        logits = model(x)
        loss = cropss_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        Gradient_cliping(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_interval == 0:
            pbar.set_postfix({
                "Step": step,          
                "Loss": f"{loss.item():.4f}", 
                "LR": f"{lr:.2e}"
            })

        if (step > 0 and step % args.save_interval == 0) or step == args.max_iters - 1:
            save_path = os.path.join(args.out_dir, f"checkpoint_step_{step}.pt")
            save_checkpoint(model, optimizer, step, save_path)

    endtime = time.time()
    cost = endtime - starttime
    m, s = divmod(cost, 60)
    h, m = divmod(m, 60)

    print(f'Done, cost : {h} hours and {m} minutes and {s} sceonds')


if __name__ == "__main__":
    train()