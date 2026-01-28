# Discrete LLM AMC

This repository implementation code for paper [DiSC-AMC: Token- and Parameter-Efficient Discretized Statistics In-Context Automatic Modulation Classification](https://arxiv.org/abs/2510.00316) a framework for Automatic Modulation Classification (AMC) using Discrete Large Language Models (LLMs) and Vision Transformers. It explores the use of advanced techniques like DINO-based autoencoders for feature extraction from signal spectrograms and finetuned LLMs for classification tasks.

## Features
- **Signal Processing**: Conversion of modulation signals into spectrograms.
- **Autoencoder Training**: Implementation of Vision Transformers (ViT) and DINO for unsupervised feature learning on spectrograms.
- **LLM Integration**: Finetuning and inference of Large Language Models (e.g., GPT-OSS, DeepSeek, Gemma) using [Unsloth](https://github.com/unslothai/unsloth) for efficient training.
- **Modulation Classification**: Classifiers based on extracted features and LLM reasoning.

## Installation

### Prerequisites
- Linux Environment
- NVIDIA GPU with CUDA support (Project setup assumes CUDA 12.8)
- Python 3.10 or higher

### Setup

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd discrete-llm-amc
    ```

2.  **Install Dependencies**:
    You can use the provided installation script to set up the environment and install necessary libraries:
    ```bash
    chmod +x install_dep.sh
    ./install_dep.sh
    ```

    Alternatively, this project uses [Poetry](https://python-poetry.org/) for dependency management. You can find the dependencies in `pyproject.toml`.

## Project Structure

```
discrete-llm-amc/
├── src/                        # Source code modules
│   ├── spectrogram/            # Spectrogram generation, DINO/ViT training scripts
│   ├── finetune/               # LLM finetuning logic (Unsloth integration)
│   ├── modulation_classification/ # Core classification algorithms
│   ├── evaluation/             # Evaluation metrics and response parsing
│   ├── data/                   # Data loading utilities
│   └── embbeding/              # Embedding utilities
├── notebooks/                  # Jupyter notebooks for experiments
│   ├── llm_inference.ipynb     # LLM loading and inference examples
│   ├── spectogram.ipynb        # Spectrogram visualization and processing
│   ├── autoencoder.ipynb       # Autoencoder experimentation
│   └── classifier.ipynb        # Classifier training and testing
├── data/                       # Dataset directories (e.g., RadioML)
├── models/                     # Checkpoints and downloaded Unsloth models
├── exp/                        # Experiment results and logs
└── papers/                     # Related literature
```

## Usage

### 1. Data Preparation
Ensure your signal data (e.g., RadioML dataset) is placed in the `data/` directory.

### 2. Spectrogram & Autoencoder
Use the scripts in `src/spectrogram/` or the `notebooks/spectogram.ipynb` and `notebooks/autoencoder.ipynb` to process signals into spectrograms and train feature extractors.

### 3. LLM Finetuning & Inference
-   **Finetuning**: Scripts in `src/finetune/` allow for finetuning models.
-   **Inference**: Check `notebooks/llm_inference.ipynb` for examples on how to load quantized models using Unsloth and perform inference.

## Key Libraries
-   **Unsloth**: For faster and memory-efficient LLM finetuning.
-   **vLLM**: For high-performance inference.
-   **PyTorch**: Deep learning framework.
-   **Torchaudio/Torchvision**: Audio and vision processing.

## License
[Insert License Here]
