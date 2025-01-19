import os
import time
import numpy as np
import torch
from model import GPTConfig, GPT
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm  # Import tqdm for progress bars

if __name__ == "__main__":
    # Variables and flags declaration (same as before)
    gradient_accumulation_steps = 4* 8
    learning_rate = 6e-4
    dataset = 'openwebtext'
    init_from = 'resume'
    max_iterations = 5000
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    min_learning_rate = 6e-5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluation_iterations = 5000
    evaluation_interval = 100
    n_layer = 10
    n_head = 8
    n_embd = 768
    vocab_size = 50304
    block_size = 256
    batch_size = 18
    dropout = 0.1
    lr_decay_iters = 5000
    dtype = 'float16'
    out_dir = 'output'
    eval_only = False
    always_save_checkpoint = True
    decay_lr = True
    grad_clip = 1.0

    # Create DataLoader for training and validation datasets
    data_dir = os.path.join('data', dataset)
    def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
        if split == 'train':
            data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
    iter_num = 0
    local_iter_num = 0
    best_val_loss = 1e9

    # Model initialization (same as before)
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, dropout=dropout, vocab_size=vocab_size)
    if init_from == 'scratch':
        print("Initializing a new model from scratch")
        config = GPTConfig(**model_args)
        model = GPT(config)
    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        config = GPTConfig(**model_args)
        model = GPT(config)
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']

    # Apply gradient checkpointing (same as before)
    model.gradient_checkpointing = True
    model.to(device)
    scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)

    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None

    # Evaluation function with tqdm progress bar
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(evaluation_iterations)  # Ensure we don't exceed the batch count
            # Use tqdm to show the progress
            for k in tqdm(range(evaluation_iterations), desc=f"Evaluating {split}"):
                X, Y = get_batch(split)
                X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device, dtype=torch.float16):
                    logits, loss = model(X, Y)
                if k < losses.size(0):
                    losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # Learning rate scheduling (same as before)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=max_iterations,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=1e4,
    )

    # Main training loop with tqdm for training progress
while iter_num < max_iterations:
    # Evaluation
    if iter_num % evaluation_interval == 0:
        if local_iter_num == 0:
            print(f"From checkpoint, step: {iter_num} validation loss:{best_val_loss:.4f}")
        else:
            print("Beginning model evaluation!")
            t2 = time.time()
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num >= 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
            t3 = time.time()
            print(f"time taken to evaluate and save the model: {(t3 - t2) / 60.0:.2f} min")

    # Training step with gradient accumulation and tqdm for progress
    t0 = time.time()  # Start time for this iteration

    for micro_step in tqdm(range(gradient_accumulation_steps), desc="Training"):
        X, Y = get_batch('train')
        X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device, dtype=torch.float16):
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        # Gradient scaling for mixed precision
        scaler.scale(loss).backward()

        # After processing the batch, continue to next one (async loading)
        if micro_step + 1 == gradient_accumulation_steps:
            # Perform the optimizer step after accumulating gradients
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            # step the scheduler
            scheduler.step()

    # Calculate time for the full iteration (not just micro-steps)
    t1 = time.time()  # End time for this iteration
    if iter_num % 5 == 0:
        print(f"iteration: {iter_num} completed in :{(t1 - t0) / 60.0:.2f} min")  # Convert time to minutes
        t0 = t1  # Reset start time for the next iteration

    iter_num += 1
    local_iter_num += 1

    # Termination condition
    if iter_num > max_iterations:
        break
