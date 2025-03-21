# PyTorch LeNet-5 Implementation for MNIST

This repository contains a PyTorch implementation of the classic LeNet-5 Convolutional Neural Network (CNN) for handwritten digit classification using the MNIST dataset.  It's designed as a tutorial, guiding you through the key concepts of CNNs and providing a fully functional example.

**Copyright (c)** 2019 Angelo Cangelosi, MIT License. Code and examples adapted from Gulli & Pal (2017) Deep Learning with Kera. Punkt Publishing

## Overview

This project is based on the pioneering work of Yann LeCun and colleagues on Convolutional Networks.  The LeNet-5 architecture is a foundational CNN, demonstrating the power of convolutional layers, pooling layers, and fully connected layers for image recognition.  The code closely follows the original LeNet architecture, with adaptations for modern PyTorch practices.

## Notebook Contents

The core of this repository is the Jupyter Notebook, `PyTorch_LeNet_MNIST.ipynb`, which covers the following:

1.  **Introduction to CNNs:** A brief explanation of the core concepts behind Convolutional Neural Networks, including:
    *   Convolutional Layers (Receptive Fields, Kernel Size, Stride, Padding)
    *   Pooling Layers (Max Pooling, Average Pooling)
    *   Feature Maps
    *   Activation Functions (ReLU)
    *   Fully Connected Layers
    *   Softmax Output

2.  **LeNet-5 Architecture:** A detailed description of the specific LeNet-5 architecture used in this implementation, including the number of layers, filter sizes, and activation functions.

3.  **PyTorch Implementation:** Step-by-step code implementation using PyTorch, covering:
    *   Importing necessary libraries (PyTorch, NumPy, Matplotlib)
    *   Defining the `LeNet` class (using `nn.Module`)
    *   Defining the convolutional and pooling layers (`nn.Conv2d`, `nn.MaxPool2d`, `nn.AvgPool2d`)
    *   Defining the fully connected layers (`nn.Linear`)
    *   Implementing the `forward` pass
    *   Loading and preprocessing the MNIST dataset (`torchvision.datasets.MNIST`, `DataLoader`)
    *   Splitting the data into training, validation, and test sets
    *   Setting up the optimizer (`optim.Adam`) and loss function (`nn.CrossEntropyLoss`)
    *   Training the model (with detailed training loop)
    *   Evaluating the model on the test set
    *   Visualizing training progress (loss and accuracy curves)
    *   Saving and loading the model (`torch.save`, `torch.load`)
    *   Implementing checkpoints
    *   Using TensorBoard for visualization (optional, but highly recommended)

4.  **Results and Analysis:**  Discussion of the training results, including achieved accuracy and potential areas for improvement.

## Getting Started

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/YOUR_GITHUB_USERNAME/pytorch-lenet-mnist.git
    cd pytorch-lenet-mnist
    ```
    (Replace `YOUR_GITHUB_USERNAME` with your actual GitHub username.)

2.  **Install Dependencies:**

    ```bash
    pip install torch torchvision numpy matplotlib
    ```
    It is recommended to install the dependencies into virtual enviroment:
    ```bash
    pip install virtualenv
    virtualenv venv
    source venv/bin/activate #for Linux and MacOS
    venv\Scripts\activate #for Windows
    pip install torch torchvision numpy matplotlib
    ```
3.  **Run the Notebook:**

    ```bash
    jupyter notebook PyTorch_LeNet_MNIST.ipynb
    ```
    or
    ```bash
    jupyter lab
    ```
    This will open the notebook in your web browser.  You can then run the code cells sequentially to train and evaluate the model.

## Using TensorBoard (Optional)

The notebook includes code to log training metrics to TensorBoard. To view these logs:

1.  **Install TensorBoard:**

    ```bash
    pip install tensorboard
    ```

2.  **Run TensorBoard:**

    ```bash
    tensorboard --logdir=logs
    ```

    This will start a TensorBoard server, and you can access it in your web browser (usually at `http://localhost:6006`).

## Credits and License

**Copyright (c) 2019 Angelo Cangelosi, MIT License.**

This code is adapted from examples in "Deep Learning with Keras" by Antonio Gulli and Sujit Pal (Packt Publishing, 2017), and is inspired by the original work:

Y. LeCun and Y. Bengio, "Convolutional Networks for Images, Speech, and Time-Series," in *The Handbook of Brain Theory and Neural Networks*, M. A. Arbib, Ed. Cambridge, MA: MIT Press, 1995.

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome!  If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
