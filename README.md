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

The project is based on a light-weight model and you can easily run it on on Google-Colab. However, you can also download the local copy of the repository to perfor experiments locally.

To use the code in this repository, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Shujaat123/ACP-LSTM-NFR.git
   cd ACP-LSTM-NFR

2. In Jupter-installed environment load the ipybn file and install required packages as prompted.

3. Make sure you have access to the following files:
  - Dataset files (ACP344.txt and ACP740.txt) in the data/ directory
  - Pretrained model weights in the best_model/ directory

# Requirements
- Python 3.7+
- Keras 2.4.3
- TensorFlow 2.x
- Pandas
- NumPy
- Openpyxl (for Excel export)
- Jupyter Notebook

# Files and Folders

## 1. `Best_model/`:
Contains pre-trained weights of the best LSTM model.

## 2. `Data/`:
Contains the dataset used in the study.

## 3. `Notebooks`:
- **ACP_LSTM_NFP.ipynb**: Independent inference notebook. Users can upload their peptide sequences in FASTA format (.txt file) for classification. The model outputs the probabilities of the sequences being ACPs and allows users to download results in an Excel file.
- **training_and_evalution.ipynb**: Contains the code for training and evaluating the model on the dataset.

## 4. `Utilities`:
- **feature_extraction.py**: Code for extracting features used during training.
- **feature_extraction_inference.py**: Utility code for feature extraction during inference.


# Usage

## Training and Evaluation
To train the model and evaluate its performance, open the `training_and_evaluation.ipynb` notebook and run the cells. The training process includes:

- Feature extraction from peptide sequences.
- Model training with LSTM architecture.
- Cross-validation and evaluation metrics.

> You can modify parameters such as the number of epochs, batch size, or use your dataset in place of the provided datasets.

## Inference
To use the model for inference, open the `ACP_LSTM_NFP.ipynb` notebook. This notebook allows users to upload a `.txt` file containing peptide sequences in FASTA format and obtain the predicted probabilities of the sequences being ACPs.

- Upload your FASTA sequences in a `.txt` file (e.g., `sample_seq.txt`).
- The model will predict the probabilities of being ACP and non-ACP.
- Results will be available for download as an Excel file.

## Feature Extraction
The repository includes two Python files that handle feature extraction:

- **feature_extraction.py**: Contains the code for extracting features from peptide sequences during training.
- **feature_extraction_inference.py**: Contains the code for feature extraction used during inference.

The feature sets include:

- Binary Profile.
- $k$-mer Sparse Matrix.
- Composition of the K-Spaced Side Chain Pairs (CKSSCP).
- Composition of the K-Spaced Electrically Charged Side Chain Pairs (CKSECSCP).
- Isoelectric Point Feature.

## Results
The model was trained using the ACP344 and ACP740 datasets and achieved state-of-the-art performance in terms of accuracy and MCC (Matthew's Correlation Coefficient). The trained models can be used to classify new peptide sequences with high reliability.

## Citing
If you use this repository or the provided datasets and models in your research, please cite the following paper: \\

Nazer Al Tahifah, Muhammad Sohail Ibrahim, Erum Rehman, Naveed Ahmed, Abdul Wahab, Shujaat Khan. "Anticancer Peptides Classification Using Long-Short-Term Memory with Novel Feature Representation " IEEE Access, Volume X(X), December 2024, DOI: 10.1109/ACCESS.2024.3523068, online  https://ieeexplore.ieee.org/document/10816412
```latex
@article{your_paper_citation,
  title={Anticancer Peptides Classification Using Long-Short-Term Memory with Novel Feature Representation},
  author={Nazer Al Tahifah, Muhammad Sohail Ibrahim, Erum Rehman, Naveed Ahmed, Abdul Wahab, and Shujaat Khan},
  journal={IEEE Access},
  year={2024},
  doi={10.1109/ACCESS.2024.3523068}}
