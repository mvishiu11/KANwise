# KANwise - Kolmogorov–Arnold Networks (KANs) Implementation

## General Overview
This project implements Kolmogorov–Arnold Networks (KANs), a novel approach to neural network architecture inspired by the Kolmogorov–Arnold representation theorem. Unlike traditional neural networks that use predefined activation functions, KANs employ learnable activation functions, providing a flexible and potentially more interpretable model. This implementation brings the theoretical elegance of KANs into practical, applied machine learning using PyTorch, making it accessible and adaptable to both research and industry applications.

For a deeper understanding, see the original paper: [KAN: Kolmogorov–Arnold Networks](https://arxiv.org/abs/2404.19756)

## Table of Contents
1. [Theoretical Introduction](#theoretical-introduction)
2. [Implementation Choices](#implementation-choices)
3. [Examples](#examples)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Contributing](#contributing)
7. [License](#license)
8. [Acknowledgments](#acknowledgments)

## Theoretical Introduction
Kolmogorov–Arnold Networks are based on the mathematical foundation provided by the Kolmogorov–Arnold representation theorem, which states that any continuous function of several variables can be represented as a superposition of continuous functions of one variable. In the context of neural networks, this implies a model where the complexity and capacity can be controlled more directly through the manipulation of these univariate functions, rather than layer depth or neuron count.

### Why is KAN Cool?
KANs offer a new perspective on neural network design:
- **Interpretability**: Each activation function has a direct mathematical interpretation.
- **Flexibility**: Custom activation functions can adapt to specific features of the input data.
- **Efficiency**: Potential reductions in the number of necessary parameters and computational resources.

## Implementation Choices
### Framework
- **PyTorch**: Chosen for its flexibility and dynamic computation graph, which fits well with the need for customizable activation functions.
- **Keras-like API**: Provides a familiar and easy-to-use interface for model definition and training, making it accessible to users transitioning from Keras.

### Architecture
- **Custom Layer**: `KANLayer`, which supports flexible spline-based activation functions.
- **Modular Design**: Allows easy integration into existing PyTorch workflows and scalability to different types of neural network architectures.

## Examples
This repository includes several examples demonstrating the practical applications of KANs:
- **Simple Regression**: Illustrates how KANs can be used for a basic regression problem.
- **PDE Solving**: Shows the application of KANs to solve partial differential equations, highlighting their potential in scientific computing.
- **Function Discovery**: Demonstrates the capability of KANs to discover underlying mathematical relationships from data.

## Installation
To install the necessary dependencies, run:
```bash
poetry install
```

Ensure you have Python 3.11+ and Poetry installed on your system.

## Usage
To run the examples use the provided scripts. For example:
```bash
poetry run simple_regression
```
Replace `simple_regression` with the script corresponding to your desired application as defined in the [examples](#examples) section.

## Contributing
Contributions to this project are welcome! Please fork the repository, make your changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments
- Thanks to the authors of the original paper on Kolmogorov–Arnold Networks for inspiring this implementation.
- Thanks to the PyTorch community for providing an excellent platform for developing such innovative machine learning tools.
