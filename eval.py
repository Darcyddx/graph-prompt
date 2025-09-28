import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import settings
from utils import (get_network, 
                   get_cifar10_test_dataloader,
                   get_cifar100_test_dataloader,
                   get_tiny_imagenet_test_dataloader)


def get_test_loader(dataset_name, args):
    config = settings.get_dataset_config(dataset_name)
    mean = config['mean']
    std = config['std']
    
    if dataset_name == 'cifar10':
        test_loader = get_cifar10_test_dataloader(
            mean, std,
            num_workers=8,
            batch_size=args.batch_size,
            shuffle=False
        )
    elif dataset_name == 'cifar100':
        test_loader = get_cifar100_test_dataloader(
            mean, std,
            num_workers=8,
            batch_size=args.batch_size,
            shuffle=False
        )
    elif dataset_name == 'tiny_imagenet':
        if any(keyword in args.net.lower() for keyword in ['vit', 'swin', 'mobilevit', 'vim']):
            img_size = 224
        else:
            img_size = 64
            
        data_root = os.path.join('./data', 'tiny')
        
        test_loader = get_tiny_imagenet_test_dataloader(
            mean, std,
            num_workers=8,
            batch_size=args.batch_size,
            shuffle=False,
            data_root=data_root,
            img_size=img_size
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return test_loader


@torch.no_grad()
def evaluate_model(net, test_loader, device, args):  
    net.eval()
    
    test_loss = 0.0
    correct = 0.0
    total = 0
    
    loss_function = nn.CrossEntropyLoss()
    
    start_time = time.time()
    
    for images, labels in test_loader:
        if device.type == 'cuda':
            images = images.cuda()
            labels = labels.cuda()
        
        outputs, _ = net(images, [0] * (args.num_elements + 1))
        loss = loss_function(outputs, labels)
        
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    end_time = time.time()
    
    accuracy = 100.0 * correct / total
    avg_loss = test_loss / len(test_loader)
    
    return accuracy, avg_loss, end_time - start_time

# Then update the function call in main:
accuracy, avg_loss, eval_time = evaluate_model(net, test_loader, device, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='cifar10', 
                        choices=['cifar10', 'cifar100', 'tiny_imagenet'],
                        help='dataset to use for evaluation')
    parser.add_argument('-net', type=str, default="mobilenet", help='network architecture')
    parser.add_argument('-num_elements', type=int, default=6, help='number of model elements (same as training)')
    parser.add_argument('-weights_path', type=str, required=True, help='path to model weights')
    parser.add_argument('-batch_size', type=int, default=128, help='batch size for evaluation')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.weights_path):
        print(f"Error: Weights file not found at {args.weights_path}")
        sys.exit(1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        dataset_config = settings.get_dataset_config(args.dataset)
        num_classes = dataset_config['num_classes']
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    net = get_network(args)
    
    try:
        state_dict = torch.load(args.weights_path, map_location=device)
        net.load_state_dict(state_dict)
        print(f"Successfully loaded weights from {args.weights_path}")
    except Exception as e:
        print(f"Error loading weights: {e}")
        sys.exit(1)
    
    test_loader = get_test_loader(args.dataset, args)
    
    print(f"Dataset: {args.dataset}")
    print(f"Network: {args.net}")
    print(f"Device: {device}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Batch size: {args.batch_size}")
    print("-" * 50)
    
    accuracy, avg_loss, eval_time = evaluate_model(net, test_loader, device)
    
    print(f"Test Accuracy: {accuracy:.2f}%")
