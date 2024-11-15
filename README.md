# Lane Detection Training with Distributed and Centralized Architectures

This repository contains Python scripts and modules for training a lane detection model using both **centralized (CPU)** and **distributed (Dual GPU)** architectures. The project is built to work with a large dataset, and the training process is structured to log detailed outputs, including metrics such as **training time**, **losses**, and **accuracies**.

### Explanation of Key Files

- **data/datasets/segmentation_dataset.py**: Contains the dataset class to load and preprocess lane detection data from CARLA Simulator.
- **models/fpn.py**: Defines the Feature Pyramid Network (FPN) model architecture used for lane segmentation.
- **utils/metrics.py**: Utility functions to calculate training and validation metrics such as accuracy and loss.
- **utils/train_helpers.py**: Helper functions for setting up the training pipeline, managing checkpoints, etc.
- **utils/centralized_training.py**: Script for training the model on a single CPU or GPU in a centralized architecture.
- **utils/distributed_training.py**: Script for training the model on a dual-GPU setup using PyTorch Distributed Data Parallel (DDP).

## Data Requirements

Due to the large size of the dataset, the input data has not been uploaded to this repository. Please download the **lane detection dataset for CARLA Driving Simulator** from Kaggle using the link below:

[Kaggle Dataset: Lane Detection for CARLA Driving Simulator](https://www.kaggle.com/datasets/thomasfermi/lane-detection-for-carla-driving-simulator)

Once downloaded, place the dataset files in the `data/` directory, or update the file paths in `segmentation_dataset.py` accordingly.

## How to Run

### Prerequisites

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/lane-detection-training.git
   cd lane-detection-training
2. pip install -r requirements.txt
3. python utils/centralized_training.py
4. # Run Distributed Training (Dual GPU)

To run distributed training on a dual-GPU setup, use the following command. This script leverages PyTorch’s Distributed Data Parallel (DDP) to scale training across multiple GPUs, significantly reducing training time and increasing efficiency.

```bash
python -m torch.distributed.launch --nproc_per_node=2 --use_env utils/distributed_training.py

### Explanation of the Command

- **`python -m torch.distributed.launch`**: This is the PyTorch module used to launch distributed training across multiple processes.
- **`--nproc_per_node=2`**: Specifies that two processes (GPUs) are used on this node (machine). Adjust this based on the number of GPUs you have.
