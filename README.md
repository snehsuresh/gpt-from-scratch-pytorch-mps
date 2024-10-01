# GPT Language Model Training

This project implements a simple GPT-based language model using PyTorch. The model is trained on the OpenWebText dataset and includes multi-head self-attention mechanisms with transformer blocks. The goal is to train the model to predict the next token in a sequence of text.

## Features
- GPT architecture with positional and token embeddings.
- Multi-head self-attention mechanism.
- Layer normalization and feed-forward network.
- Training on large text datasets with batch sampling.
- Generation of text sequences based on a context input.
- Efficient file handling using memory mapping for large datasets.

## Requirements
- Python 3.x
- PyTorch
- mmap
- pickle
- argparse

## Installation
1. Clone this repository:
    ```
    git clone <repository_url>
    ```
2. Install the required Python packages:
    ```
    pip install -r requirements.txt
    ```

## Usage

### Training the Model
To train the model, run the `training.py` script with the required batch size parameter:

```bash
python training.py -batch_size <value>
```

- Initialize and train the GPT model.
- Periodically print the training and validation loss.
- Save the trained model as a `.pkl` file.

### Text Generation
Once the model is trained, you can generate text sequences by using the `generate` function in the `GPTLanguageModel` class. Provide a starting sequence and specify the number of tokens to generate.

## Project Structure
- **training.py**: Main script for training the model and saving it.
- **openwebtext/**: Folder containing the dataset splits (`train_split.txt`, `val_split.txt`) and `vocab.txt`.
- **model-01.pkl**: Saved model file after training.
- **requirements.txt**: List of required Python packages.

## Dataset
The model is trained on the OpenWebText dataset. Make sure to download and preprocess the dataset before training. The text files should be split into training and validation sets, which are then used for sampling during training.

## Configuration
The following hyperparameters can be adjusted in `training.py`:
- `batch_size`: Number of samples per training batch.
- `block_size`: The context window size for each batch.
- `max_iters`: Total number of training iterations.
- `n_embd`, `n_head`, `n_layer`: Model architecture parameters (embedding size, number of attention heads, number of transformer layers).
- `dropout`: Dropout rate for regularization.
- `learning_rate`: Learning rate for the optimizer.

## Evaluation
The script periodically evaluates the training and validation loss every 100 iterations to monitor performance. Use the `estimate_loss()` function to compute the average loss over a number of evaluation steps.

## Model Saving and Loading
After training, the model is saved in a `.pkl` file using Python's `pickle` module. You can load the model for inference or further training by unpickling the file.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
This project is inspired by the original GPT architecture and its implementation in PyTorch. Special thanks to OpenWebText for providing the dataset.



