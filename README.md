# DeepAR 

A minimalistic, from-scratch implementation of the DeepAR (Probabilistic Forecasting with Autoregressive Recurrent Networks) model using PyTorch.

## Project Structure

- `src/`: Core logic
  - `model.py`: LSTM-based architecture with Gaussian likelihood head.
  - `dataset.py`: Sliding-window time-series grouping and loading logic.
  - `utils.py`: Gaussian Negative Log-Likelihood (NLL) implementation.
- `scripts/`: Execution scripts
  - `train_model.py`: Synthetic sine-wave data generation and training loop.

## Setup & Training

1.  **Environment**: Ensure you are using `numpy<2` (1.26.x) for compatibility with modern PyTorch builds.
2.  **Train**: Run the training loop with synthetic sine wave data:
    ```bash
    cd scripts
    python3 train_model.py
    ```

## Logic Overview

The model takes a window of past observations to predict the parameters ($\mu, \sigma$) of a Gaussian distribution for future time steps. The training objective is to minimize the Negative Log-Likelihood of the actual observed values under the predicted distributions.
