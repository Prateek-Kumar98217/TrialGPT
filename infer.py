import os
import torch
import tiktoken
from model import GPTConfig, GPT

# Inference configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
out_dir = 'output'  # Output directory where the model checkpoints are saved
checkpoint_path = os.path.join(out_dir, 'ckpt.pt')  # Path to the saved checkpoint
vocab_size = 50304  # The vocabulary size of the model (same as in the training script)
block_size = 256  # The block size (same as in the training script)
temperature = 1.0  # Controls the randomness of predictions (higher value = more random)
top_k = 10 # Number of top-k tokens to sample from
max_new_tokens = 300  # Number of tokens to generate in the inference process

# Initialize the tokenizer (use GPT-2 tokenizer since it's similar to your model's)
tokenizer = tiktoken.get_encoding("gpt2")

# Load the model from the checkpoint
print(f"Loading model from checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)
model_args = checkpoint['model_args']
config = GPTConfig(**model_args)
model = GPT(config)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()  # Set the model to evaluation mode

# Preprocess the input text
def encode_input_text(input_text, tokenizer, block_size):
    # Tokenize input text with tiktoken tokenizer
    input_ids = tokenizer.encode(input_text)
    # Ensure the sequence length doesn't exceed block_size
    if len(input_ids) > block_size:
        input_ids = input_ids[:block_size]
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    return input_ids

# Example input text to prompt the model
input_text = "Once upon a time, long ago ..."

# Encode the input text
input_ids = encode_input_text(input_text, tokenizer, block_size)

# Perform text generation using the generate method
generated_ids = model.generate(input_ids, max_new_tokens, temperature=temperature, k_top=top_k)

# Decode the generated token IDs to text
generated_text = tokenizer.decode(generated_ids[0].cpu().numpy())

print(f"Generated text:\n{generated_text}")
