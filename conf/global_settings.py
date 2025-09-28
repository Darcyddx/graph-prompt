import os
from datetime import datetime

# CIFAR-10 dataset settings
CIFAR10_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_TRAIN_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_NUM_CLASSES = 10

# CIFAR-100 dataset settings
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
CIFAR100_NUM_CLASSES = 100

# Tiny ImageNet dataset settings
TINY_IMAGENET_PATH = './data'
TINY_IMAGENET_TRAIN_MEAN = (0.4816396, 0.44874296, 0.3988555)
TINY_IMAGENET_TRAIN_STD = (0.22924888, 0.2262128, 0.22626047)
TINY_IMAGENET_NUM_CLASSES = 200

# Default dataset selection (can be overridden by command line arguments)
DEFAULT_DATASET = 'cifar100'

# Dataset configurations mapping
DATASET_CONFIGS = {
    'cifar10': {
        'num_classes': CIFAR10_NUM_CLASSES,
        'mean': CIFAR10_TRAIN_MEAN,
        'std': CIFAR10_TRAIN_STD
    },
    'cifar100': {
        'num_classes': CIFAR100_NUM_CLASSES,
        'mean': CIFAR100_TRAIN_MEAN,
        'std': CIFAR100_TRAIN_STD
    },
    'tiny_imagenet': {
        'num_classes': TINY_IMAGENET_NUM_CLASSES,
        'mean': TINY_IMAGENET_TRAIN_MEAN,
        'std': TINY_IMAGENET_TRAIN_STD,
        'data_path': TINY_IMAGENET_PATH
    }
}

# Training settings
EPOCH = 200
MILESTONES = [60, 120, 160]

# Directory settings
CHECKPOINT_PATH = 'checkpoint'
LOG_DIR = 'runs'

# Save settings
SAVE_EPOCH = 10

# Time settings
DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

# Helper function to get dataset configuration
def get_dataset_config(dataset_name):
    """
    Get configuration for specified dataset
    
    Args:
        dataset_name: str, one of 'cifar10', 'cifar100', 'tiny_imagenet'
    
    Returns:
        dict: configuration dictionary for the dataset
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset {dataset_name} not supported. Available: {list(DATASET_CONFIGS.keys())}")
    
    return DATASET_CONFIGS[dataset_name]