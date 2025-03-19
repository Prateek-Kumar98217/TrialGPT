# TrailGPT

TrailGPT is a project that involves training, evaluating, and using a GPT model for text generation. This repository contains scripts for training the model, preprocessing the dataset, and performing inference.

## Files

### `train.py`
This script is used for training the GPT model. It includes:
- Loading and preprocessing the dataset.
- Initializing the model and optimizer.
- Training the model with gradient accumulation.
- Evaluating the model periodically.
- Saving the model checkpoints.

### `model.py`
This script defines the architecture of the GPT model. It includes:
- `MultiLayerPerceptron`: A multi-layer perceptron used in the model.
- `CausalSelfAttention`: The self-attention mechanism.
- `Block`: A transformer block consisting of layer normalization, self-attention, and a perceptron.
- `GPTConfig`: Configuration class for the GPT model.
- `GPT`: The main GPT model class with methods for forward pass and text generation.

### `infer.py`
This script is used for generating text using the trained GPT model. It includes:
- Loading the model from a checkpoint.
- Encoding input text.
- Generating text based on the input prompt.
- Decoding the generated tokens to text.

### `preprocess.py`
This script is used for preprocessing the dataset. It includes:
- Loading the dataset from Hugging Face.
- Splitting the dataset into training and validation sets.
- Tokenizing the dataset using the GPT-2 tokenizer.
- Saving the tokenized dataset to binary files for efficient loading during training.

## Usage

### Virtual Environment
creating and activating a virtual environment is recommended before running any scripts:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Installation
Install the required packages using:
```bash
pip install -r requirements.txt
```

### Data Preprocessing
To preprocess the dataset, run:
```bash
python preprocess.py
```

### Model Training
To train the model, run:
```bash
python train.py
```

### Model Inference
To generate text using the trained model, run:
```bash
python infer.py
```

## Requirements
- Python 3.8+

## License
This project is licensed under the MIT License.

## Acknowledgements
- The GPT model architecture is inspired by OpenAI's GPT-2.
- The dataset used is OpenWebText, available on Hugging Face.
