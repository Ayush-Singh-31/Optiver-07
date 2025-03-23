# Volatility Predictor

A lightweight tool for forecasting stock volatility using ultra-high-frequency data. The project uses financial order book data to compute key metrics, apply time series models, and generate interpretable predictions to help traders make informed decisions.

## Overview

- **Objective:** Predict future volatility for over 100 stocks by processing real trading data.
- **Key Challenges:**
  - Handling ultra-high-frequency data
  - Choosing the right modeling approach (realized volatility vs. direct stock price modeling)
  - Evaluating models using multiple performance measures
- **Outcome:** A prediction tool that not only forecasts volatility but also provides clear insights into underlying market dynamics.

## Requirments

- Python 3.x
- Required libraries:

## Features

- **Data Processing:**

  - Cleans raw order book data by removing unstable initial days.
  - Forward-fills missing values to maintain time-series continuity.
  - Uses a purged group time-series split with a gap to prevent future data leakage.
  - Converts volatility targets into a multi-label classification problem with sample weighting.

- **Model Architecture:**

  - **Supervised Autoencoder:** Compresses input data into a low-dimensional latent space while learning features relevant for volatility prediction.
  - **MLP Classifier:** Combines original inputs with latent features to predict high-volatility events.
  - **Joint Training:** Ensures that feature extraction and classification are optimized simultaneously, reducing the risk of leakage.

- **Regularization & Optimization:**
  - Incorporates dropout, batch normalization, and noise injection to improve generalization.
  - Uses the swish activation function for smoother gradients.
  - Supports hyperparameter tuning (e.g., via Hyperopt) and training with multiple random seeds for stability.

## Approach

### Data Preprocessing & Cross-Validation

- **Cleaning:**

  - Remove the first few days to eliminate unstable trading data.
  - Forward-fill missing values to preserve the sequence.

- **Time-Series Split:**
  - Use a purged group time-series split with a set gap (e.g., 31 days) to avoid look-ahead bias.
  - Transform volatility targets into a multi-label classification format.
  - Weight samples by the absolute market response to highlight significant events.

### Model Architecture

#### Supervised Autoencoder

- **Encoder:**
  - Maps the input vector \( x \) to a lower-dimensional latent representation \( z \) using non-linear activation (e.g., swish).
- **Decoder:**
  - Reconstructs the original input from \( z \) to measure how well the latent space captures the data structure.
- **Supervised Training:**
  - The latent features are optimized with both a reconstruction loss and a classification loss, ensuring relevance to the volatility prediction task.

#### MLP Classifier

- Processes a concatenation of the original features and the latent representation \([x, z]\).
- Consists of several layers with non-linear activations, designed to predict high-volatility events accurately.

### Loss Function & Joint Training

- **Loss Composition:**
  - Overall loss: \( L = \alpha L*{\text{recon}} + \beta L*{\text{cls}} \)
- **Reconstruction Loss \( L\_{\text{recon}} \):**
  - Measures the difference between the original input and its reconstruction.
- **Classification Loss \( L\_{\text{cls}} \):**
  - Uses binary cross-entropy (summed across multiple volatility thresholds) to guide the classifier.
- **Joint Backpropagation:**
  - Both the autoencoder and MLP classifier are updated together during training to ensure consistent feature learning without validation leakage.

### Additional Techniques

- **Activation & Regularization:**
  - Uses the swish activation function to avoid issues like dead neurons.
  - Applies dropout, batch normalization, and noise injection to enhance model robustness.
- **Optimization:**
  - Trains the model with multiple random seeds and optimizes hyperparameters to reduce variance and improve performance.

### Potential Improvements

- Explore sequential models (LSTM, GRU, or transformers) for richer temporal feature extraction.
- Add residual connections in the autoencoder or MLP to stabilize training.
- Combine raw features with handcrafted signals (e.g., technical indicators) for an enriched input space.
- Investigate attention mechanisms to focus on critical time periods.
- Implement early stopping and ensemble methods to boost predictive performance.
