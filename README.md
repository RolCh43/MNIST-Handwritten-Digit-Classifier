#  MNIST Handwritten Digit Classifier

This project implements a **simple neural network from scratch (no TensorFlow or PyTorch)** to recognize handwritten digits from the **MNIST dataset** using only **NumPy** and **Python standard libraries**.

It was built for an assignment, for my Artifitial Inteligence class, to understand how neural networks work internally, from reading raw binary data to training and evaluating a model.

---

##  Project Structure

ML_NUMBERS/

│

├── data/ # MNIST dataset files (.ubyte)

│ ├── train-images-idx3-ubyte

│ ├── train-labels-idx1-ubyte

│ ├── t10k-images-idx3-ubyte

│ └── t10k-labels-idx1-ubyte

│
├── dataloader.py # Handles reading MNIST binary files

├── neural_network.py # Implementation of a simple feedforward NN

├── trainer.py # Handles training loop and evaluation

├── utils.py # Helper functions (e.g. one-hot encoding)

├── main.py # Entry point – loads data, trains the model

└── README.md # Project documentation (this file)


## Setup

### 1. Clone the repository
```
git clone https://github.com/your-username/ml-numbers.git
cd ml-numbers
2. (Optional) Create a virtual environment


python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
3. Install dependencies


pip install numpy 
4. Download MNIST data
Download the following four files and place them in the data/ folder:

File	Description
train-images-idx3-ubyte	Training images
train-labels-idx1-ubyte	Training labels
t10k-images-idx3-ubyte	Test images
t10k-labels-idx1-ubyte	Test labels

You can get them from Yann LeCun’s MNIST page.

Make sure they are unzipped (no .gz extension).
```

### Run the Project
Simply run:
```
python main.py
```
The script will:

Load the MNIST dataset from binary files

Normalize and one-hot encode the data

Build and train a neural network

Print training progress and accuracy

### Code Overview
MnistDataloader
Reads binary MNIST files (.ubyte) and converts them into NumPy arrays.

NeuralNetwork
Implements a basic feedforward neural network with:

Fully connected layers

Sigmoid activation

Forward and backward propagation

Weight updates using gradient descent

Trainer
Coordinates the training process, runs epochs, computes loss, and prints accuracy.

Utils
Contains helper functions such as one-hot encoding.

### Learning Objectives
Understand how MNIST data is structured

Learn how to load and preprocess binary datasets

Implement forward/backward propagation manually

Visualize and interpret model performance

### Technologies Used
Python 3.14

NumPy


### Author
Roldão Pitra

Software Engineering @ UCSAL
