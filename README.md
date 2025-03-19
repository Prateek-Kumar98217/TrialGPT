

**TrialGPT** is a custom GPT-2-based language model designed for text generation tasks. This project involves training the model from scratch, evaluating its performance, and performing inference to generate coherent text. TrialGPT is developed to explore the workings of **Large Language Models (LLMs)** and is based on the **OpenWebText** and **Dolma v16-sample** datasets.

## Features

- **Custom GPT-2 Architecture**: Built from scratch to handle text generation tasks.
- **Training Pipeline**: Optimized training with gradient clipping, memory mapping, and cloud computing for efficient scaling.
- **Text Generation**: Generate coherent text based on input prompts.
- **Evaluation**: Assess model performance using various natural language tasks.
- **Optimizations**: Reduced training time by 20% and improved model efficiency using advanced techniques.

## Files

### 1. `train.py`
- **Purpose**: Script to train the GPT model.
- **Features**:
  - Loads and preprocesses the dataset.
  - Initializes the model and optimizer.
  - Performs training with gradient accumulation.
  - Periodically evaluates the model during training.
  - Saves model checkpoints for future use.

### 2. `model.py`
- **Purpose**: Defines the architecture of the GPT model.
- **Features**:
  - Implements a **multi-layer perceptron** (MLP) used in the model.
  - Defines **CausalSelfAttention** (the self-attention mechanism).
  - Constructs **transformer blocks** for layer normalization, self-attention, and perceptrons.
  - Provides a **GPTConfig** for model configuration.
  - Defines the main **GPT** class for the forward pass and text generation methods.

### 3. `infer.py`
- **Purpose**: Script to generate text using the trained model.
- **Features**:
  - Loads the trained model from a checkpoint.
  - Encodes input text for generation.
  - Generates text based on input prompts.
  - Decodes generated tokens into readable text.

### 4. `preprocess.py`
- **Purpose**: Preprocesses the dataset for training.
- **Features**:
  - Loads the dataset from **Hugging Face**.
  - Splits the dataset into training and validation sets.
  - Tokenizes the dataset using **GPT-2 tokenizer**.
  - Saves the tokenized dataset to binary files for efficient training.

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
