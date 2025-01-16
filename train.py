import os
import time
import numpy as np
import torch
import pickle
from model import GPTConfig, GPT
import math
# variables and flags declaration
gradient_accumulation_steps=4*8 #used to reduce number of gradient updates per iteration and simulate larger batch size
learning_rate = 6e-4
dataset = 'openwebtext' #name of dataset used or the nme of folder containing the dataset
init_from = 'resume' #can be scratch to initialize model from scratch, resume for continuing training from checkpoint
max_iterations = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
warmup_iterations = 200
min_learning_rate = 6e-5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using {device} as device")
evaluation_iterations = 5000
evaluation_interval = 300   #controls the frequency of evaluation and checkpoint saves
n_layer =10
n_head =8
n_embd =512
vocab_size =50304
block_size =512
batch_size =12
dropout =0.1
lr_decay_iters=5000
dtype = 'float16'   #used formixed percision training, if available can use float32 and bfloat16
out_dir='output'
eval_only = False  # Set to True if you want to only evaluate
always_save_checkpoint = True # Set to True if you want to save after every evaluation
decay_lr = True  # Set to False if you do not want learning rate decay
grad_clip=1.0

token_per_iteration=gradient_accumulation_steps*block_size*batch_size
print(f"Tokens used per iteration: {token_per_iteration}")
os.makedirs(out_dir, exist_ok=True)
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device, dtype=ptdtype) #used for mixed percision training
torch.manual_seed(42)
data_dir = os.path.join('data', dataset)
def get_batch(split): #manual dataloader
    # We recreate np.memmap every batch to avoid a memory leak
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
iter_num = 0  #maintains the total iteration number throughout model training
best_val_loss = 1e9 #maintaion best validation loss throughout training, set to high value for easy first update
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, dropout=dropout, vocab_size=vocab_size)
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    config = GPTConfig(**model_args)
    model = GPT(config)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    config = GPTConfig(**model_args)
    model = GPT(config)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
model.to(device)
scaler = torch.amp.GradScaler('cuda',enabled=(dtype == 'float16')) #amp scaler for mixed percision training
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None
@torch.no_grad()
def estimate_loss(): #evaluation loss calculation
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(evaluation_iterations)
        for k in range(evaluation_iterations):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
def get_lr(it): #manual learning rate calculator using cosine scheduling
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iterations:
        return learning_rate * (it + 1) / (warmup_iterations + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_learning_rate
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iterations) / (lr_decay_iters - warmup_iterations)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_learning_rate + coeff * (learning_rate - min_learning_rate)
X, Y = get_batch('train') # fetch the very first batch
local_iter_num = 0 # number of iterations in the lifetime of this process
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % evaluation_interval == 0:
        if local_iter_num == 0:   #to skip the evaluation process in the begining when resuming from checkpoint
            print(f"step {iter_num}: val loss {best_val_loss:.4f}, from previous checkpoint")
            continue
        else:
            print("Beginning model evaluation!")
            t2 = time.time()
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
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
            print(f"time taken to evaluate and save the model{(t3-t2)/60.0:.2f} min")
    if iter_num == 0 and eval_only:
        break
    t0=time.time()
    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)
    t1=time.time()
    if iter_num%5 == 0:
        print(f"iteration: {iter_num} completed in :{(t1-t0)*5/60.0:.2f}min ")
        t0 = t1
    iter_num += 1
    local_iter_num += 1
    # termination conditions
    if iter_num > max_iterations:
        break