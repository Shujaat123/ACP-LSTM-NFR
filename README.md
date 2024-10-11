# Creating a markdown string for the README file
readme_content = """
# ACP-LSTM-NFR: Anticancer Peptides Classification Using Long-Short-Term Memory with Novel Feature Representation

This repository contains the code and resources for the paper **"Anticancer Peptides Classification Using Long-Short-Term Memory with Novel Feature Representation"**. The project focuses on predicting anticancer peptides (ACPs) from peptide sequences using a Long-Short-Term Memory (LSTM)-based model with novel feature representations.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model](#model)
- [Notebooks](#notebooks)
- [Installation](#installation)
- [Usage](#usage)
  - [Training and Evaluation](#training-and-evaluation)
  - [Inference](#inference)
- [Feature Extraction](#feature-extraction)
- [Results](#results)
- [Citing](#citing)
- [License](#license)

## Overview

Cancer treatment is a complex challenge, and **anticancer peptides (ACPs)** represent a promising avenue for targeted therapies. However, identifying these peptides remains a difficult task. This project introduces an LSTM-based model trained on a novel combination of feature extraction techniques to classify ACPs.

### Key Features:
- **Five feature encoding techniques** including binary profile, $k$-mer sparse matrix, composition of K-spaced side chain pairs (CKSSCP), composition of K-spaced electrically charged side chain pairs (CKSECSCP), and isoelectric point features.
- A robust LSTM architecture trained on two datasets, ACP344 and ACP740.
- Demonstrates superior classification performance compared to other methods.

## Dataset

The `data` folder contains the datasets used for training and evaluating the model:
- **ACP344**: Contains 344 peptide sequences (138 ACPs, 206 non-ACPs).
- **ACP740**: Contains 740 peptide sequences (376 ACPs, 364 non-ACPs).

### File Structure
- `data/ACP344.txt`: Dataset for ACP344.
- `data/ACP740.txt`: Dataset for ACP740.

## Model

The LSTM-based model is trained using the feature sets extracted from peptide sequences. The model architecture is as follows:
- **LSTM layer**: 128 units with dropout regularization.
- **Dense layer**: Sigmoid activation for binary classification (ACP or non-ACP).

The best model weights are stored in the `best_model` folder:
- `best_model/model_acp740_4.keras`: Pre-trained weights for the ACP740 dataset.

## Notebooks

- **`ACP_LSTM_NFP.ipynb`**: This is the most important notebook. It contains the independent inference code. Users can upload FASTA sequences in a `.txt` file, and the model will predict the probabilities of the sequences being ACPs. Results can be downloaded as an Excel file.
- **`training_and_evaluation.ipynb`**: Contains code for training the model and evaluating its performance.
- **`feature_extraction.py`**: Utilities for feature extraction.
- **`feature_extraction_inference.py`**: Utilities for feature extraction during inference.

## Installation

To use the code in this repository, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Shujaat123/ACP-LSTM-NFR.git
   cd ACP-LSTM-NFR
