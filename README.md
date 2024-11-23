# Maximizing Sum-Rate in MIMO Systems Using Neural Network-Based Precoding

## Overview

This project investigates the use of **deep learning-based neural networks** to maximize the **sum-rate** in **multi-user multiple-input multiple-output (MU-MIMO)** communication systems. The primary goal of this project is to leverage an **Encoder-Decoder neural network** architecture to optimize precoding in MIMO systems while ensuring fairness among users.

The proposed method aims to provide a scalable and efficient alternative to traditional iterative methods like **Weighted Minimum Mean Square Error (WMMSE)**, which are computationally expensive and not suitable for large-scale systems.

## Key Features

- **Encoder-Decoder Neural Network Architecture**: The system uses a neural network to predict optimal precoding matrices for MIMO systems based on channel state information (CSI).
- **Fairness Consideration**: Implements **Proportional Fairness** to ensure that all users receive a fair share of resources, preventing any user’s sum-rate from decreasing over time.
- **Efficient Computation**: The model significantly improves computation speed compared to traditional methods by replacing iterative optimization with a parallelizable neural network approach.
- **Near-Optimal Performance**: The approach achieves near-optimal sum-rate, comparable to WMMSE, with up to **47x faster** computation.

## Project Structure

The project is organized into the following main components:
1. **Model Implementation**: Defines the Encoder-Decoder network and training procedure for optimizing the precoding matrices.
2. **Loss Function**: Uses a custom loss function to optimize for **sum-rate maximization** with a focus on **fairness** across users.
3. **Training & Evaluation**: Code for training the model, evaluating performance, and plotting results.
4. **Experiment Results**: Includes scripts for experimenting with different network parameters, user numbers, and power levels, and visualizing the sum-rate performance.

## Requirements

- Python 3.6 or higher
- PyTorch (v1.10.0 or higher)
- NumPy
- Matplotlib

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
