import os
import time
import numpy as np
import torch
from model import GPTConfig, GPT
from torch.optim.lr_scheduler import OneCycleLR  # can experiment with other lr schedulers, I find this better for smoother training
from tqdm import tqdm  # tqdm for progress bars

if __name__ == "__main__":
    # Variables and flags declaration
    gradient_accumulation_steps = 4 * 8  # This value simulates larger batches, reduce this to decrease number of forward passes before a single backward pass. Also keep this value as a multiplication of two numbers and change only the first number for easier tuning of this training parameter.
    learning_rate = 6e-4
    dataset = 'openwebtext'
    init_from = 'resume'  # Set this value to either 'scratch' or 'resume' depending on the training phase.
    max_iterations = 50000
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluation_interval = 100  # This value controls the total training iterations before each evaluation
    loss_sampling = 1000  # This value controls the number of samples taken for average loss estimation during evaluation. Or the number of time evaluation takes place before averaging out the result. Reduce this for quicker evaluation
    n_layer = 10
    n_head = 8
    n_embd = 768
    vocab_size = 50304
    block_size = 256
    batch_size = 18
    dropout = 0.1
    dtype = 'float16'
    out_dir = 'output'
    eval_only = False
    always_save_checkpoint = False  # Set this to True only if you want to save the model after evaluation, even if the loss does not decrease from the previous evaluation.
    grad_clip = 1.0

    # Data fetching and batching
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
            # pin arrays x, y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    iter_num = 0
    best_val_loss = 1e9

    # Model initialization
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, dropout=dropout, vocab_size=vocab_size)
    if init_from == 'scratch':
        print("Initializing a new model from scratch")
        config = GPTConfig(**model_args)
        model = GPT(config)
    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)  # Leave as is, without setting `weights_only=True`
        checkpoint_model_args = checkpoint['model_args']
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        config = GPTConfig(**model_args)
        model = GPT(config)
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        print(f"From checkpoint, step: {iter_num} validation loss:{best_val_loss:.4f}")

    # Apply gradient checkpointing
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
            losses = torch.zeros(loss_sampling)  # Ensure we don't exceed the batch count
            # Use tqdm to show the progress
            for k in tqdm(range(loss_sampling), desc=f"Evaluating {split}"):
                X, Y = get_batch(split)
                X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device, dtype=torch.float16):
                    logits, loss = model(X, Y)
                if k < losses.size(0):
                    losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # Learning rate scheduling
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=max_iterations,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=30.0,
        final_div_factor=1e4,
    )

    # Main training loop with tqdm for training progress
    while iter_num < max_iterations:
        # Evaluation
        if iter_num % evaluation_interval == 0 or eval_only:
            print("Beginning model evaluation!")
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
        
        if eval_only:
            break
        else:
            # Training step with gradient accumulation and tqdm for progress
            for micro_step in tqdm(range(gradient_accumulation_steps), desc="Training"):
                X, Y = get_batch('train')
                X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device, dtype=torch.float16):
                    logits, loss = model(X, Y)
                    loss = loss / gradient_accumulation_steps
                scaler.scale(loss).backward()

                if micro_step + 1 == gradient_accumulation_steps:
                    # Perform the optimizer step after accumulating gradients
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    # step the scheduler
                    scheduler.step()

            iter_num += 1

        # Termination condition
        if iter_num > max_iterations:
            break
