import os
import sys
import argparse
import time
import random
import logging
import math
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from indicator import compute_similarity_indicator
from conf import settings
from utils import (get_network, 
                   get_cifar10_training_dataloader, get_cifar10_test_dataloader,
                   get_cifar100_training_dataloader, get_cifar100_test_dataloader,
                   get_tiny_imagenet_training_dataloader, get_tiny_imagenet_test_dataloader,
                   WarmUpLR, most_recent_folder, most_recent_weights, last_epoch, best_acc_weights)
from computeweight import compute_adaptive


def get_dataset_loaders(dataset_name, args):
    """
    Get the appropriate data loaders based on dataset name
    
    Args:
        dataset_name: str, one of 'cifar10', 'cifar100', 'tiny_imagenet'
        args: command line arguments
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    config = settings.get_dataset_config(dataset_name)
    mean = config['mean']
    std = config['std']
    
    if dataset_name == 'cifar10':
        train_loader = get_cifar10_training_dataloader(
            mean, std,
            num_workers=8,
            batch_size=args.b,
            shuffle=True
        )
        test_loader = get_cifar10_test_dataloader(
            mean, std,
            num_workers=8,
            batch_size=args.b,
            shuffle=False
        )
    elif dataset_name == 'cifar100':
        train_loader = get_cifar100_training_dataloader(
            mean, std,
            num_workers=8,
            batch_size=args.b,
            shuffle=True
        )
        test_loader = get_cifar100_test_dataloader(
            mean, std,
            num_workers=8,
            batch_size=args.b,
            shuffle=False
        )
    elif dataset_name == 'tiny_imagenet':
        # Determine image size based on model type
        if any(keyword in args.net.lower() for keyword in ['vit', 'swin', 'mobilevit', 'vim']):
            img_size = 224
            logger.info(f"Using Transformer preprocessing ({args.net}): resizing to {img_size}x{img_size}")
        else:
            img_size = 64
            logger.info(f"Using standard preprocessing for {args.net}")
            
        data_root = os.path.join('./data', 'tiny')
        
        train_loader = get_tiny_imagenet_training_dataloader(
            mean, std,
            num_workers=8,
            batch_size=args.b,
            shuffle=True,
            data_root=data_root,
            img_size=img_size
        )
        test_loader = get_tiny_imagenet_test_dataloader(
            mean, std,
            num_workers=8,
            batch_size=args.b,
            shuffle=False,
            data_root=data_root,
            img_size=img_size
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return train_loader, test_loader


def train(epoch):
    start = time.time()
    net.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_graph_loss = 0.0
    n_batches = 0

    for batch_index, (images, labels) in enumerate(training_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()

        # Create state list based on the selected mode
        state_list = get_state_list(args)

        # Forward pass with specified blocks active
        outputs, graph_list = net(images, state_list)

        # Calculate CE loss
        ce_loss = loss_function(outputs, labels)

        # Calculate graph loss for specified blocks
        if len(graph_list) > 1:
            # Handle ground truth embedding generation based on dataset
            if args.dataset == 'cifar10':
                # For CIFAR-10, create one-hot ground truth embedding instead of using prediction scores
                batch_size = labels.size(0)
                num_classes = outputs.size(1)  # Get number of classes from output

                # Create one-hot encoding of ground truth labels
                ground_truth_one_hot = torch.zeros(batch_size, num_classes)
                if args.gpu:
                    ground_truth_one_hot = ground_truth_one_hot.cuda()
                ground_truth_one_hot.scatter_(1, labels.unsqueeze(1), 1)

                # Generate ground truth embedding using RelationshipLayer
                ground_truth_embedding = net.relationship_layer(ground_truth_one_hot, 1)
            else:
                # For CIFAR-100 and Tiny ImageNet, use the last graph as ground truth
                ground_truth_embedding = graph_list[-1]

            similarity_indicator = compute_similarity_indicator(labels)

            total_graph_loss_batch = 0.0
            # The last graph in graph_list is from the output layer
            # All other graphs are from the feature layers that were activated

            # Find active indices from state_list (excluding the output layer)
            active_indices = [idx for idx, is_active in enumerate(state_list[:-1]) if is_active == 1]

            # Count active graphs for equal weighting method
            num_active_graphs = len(active_indices)

            # For alpha weighting, collect individual losses first
            if args.weight_method == 'alpha':
                individual_losses = []
                for i, relationship in enumerate(graph_list[:-1]):
                    # Use detach based on args.use_detach flag
                    if args.use_detach:
                        dist = torch.norm((ground_truth_embedding * similarity_indicator).detach() - relationship,
                                          p=2) ** 2 / ground_truth_embedding.numel()
                    else:
                        dist = torch.norm((ground_truth_embedding * similarity_indicator) - relationship,
                                          p=2) ** 2 / ground_truth_embedding.numel()
                    individual_losses.append(dist)

                # Compute alpha values using the imported function
                alpha_values = compute_adaptive(torch.stack(individual_losses))

                # Apply detach to alpha values based on args.detach_alpha flag
                if args.detach_alpha:
                    alpha_values = alpha_values.detach()

                # Apply alpha values to individual losses
                total_graph_loss_batch = torch.sum(torch.stack(individual_losses) * alpha_values)
            else:
                # Original weighting methods
                for i, relationship in enumerate(graph_list[:-1]):
                    # Use detach based on args.use_detach flag
                    if args.use_detach:
                        dist = torch.norm((ground_truth_embedding * similarity_indicator).detach() - relationship,
                                          p=2) ** 2 / ground_truth_embedding.numel()
                    else:
                        dist = torch.norm((ground_truth_embedding * similarity_indicator) - relationship,
                                          p=2) ** 2 / ground_truth_embedding.numel()

                    # Get the original index from state_list
                    original_index = active_indices[i]

                    # Apply weight based on the weighting method and number of active graphs
                    weight = calculate_weight(original_index, args.num_elements, args.weight_method, num_active_graphs)
                    dist = dist * weight

                    total_graph_loss_batch += dist

            # Scale the graph loss with the weight parameter
            graph_loss = args.graph_loss_weight * total_graph_loss_batch

            # Total loss is CE loss + graph loss
            loss = ce_loss + graph_loss

            # Log graph loss
            n_iter = (epoch - 1) * len(training_loader) + batch_index + 1
            logger.info(
                f'Iteration {n_iter} - CE Loss: {ce_loss.item():.4f}, Graph Loss: {graph_loss.item():.6f}, Total Loss: {loss.item():.4f}')

            # Keep track of total losses
            total_ce_loss += ce_loss.item()
            total_graph_loss += graph_loss.item()
        else:
            # If no feature blocks were specified (only output layer)
            loss = ce_loss
            n_iter = (epoch - 1) * len(training_loader) + batch_index + 1
            logger.info(
                f'Iteration {n_iter} - CE Loss: {ce_loss.item():.4f}, Total Loss: {loss.item():.4f}')
            total_ce_loss += ce_loss.item()

        loss.backward()
        optimizer.step()

        # Track total loss for epoch average
        total_loss += loss.item()
        n_batches += 1

        n_iter = (epoch - 1) * len(training_loader) + batch_index + 1

        # Log gradient information for debugging (debug level)
        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                logger.debug(f'LastLayerGradients/grad_norm2_weights: {para.grad.norm()}, iteration: {n_iter}')
            if 'bias' in name:
                logger.debug(f'LastLayerGradients/grad_norm2_bias: {para.grad.norm()}, iteration: {n_iter}')

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tCE Loss: {:0.4f}'.format(
            loss.item(),
            ce_loss.item(),
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(training_loader.dataset)
        ), end='')

        # Print graph loss if graph loss was calculated
        if 'graph_loss' in locals():
            print('\tGraph Loss: {:0.6f}'.format(graph_loss.item()), end='')

        print('\tLR: {:0.6f}'.format(optimizer.param_groups[0]['lr']))

        # Log training loss for each iteration (to debug level)
        logger.debug(f'Train/loss: {loss.item():.4f}, iteration: {n_iter}')

        if epoch <= args.warm:
            warmup_scheduler.step()

    # Calculate and log average losses for the epoch
    avg_loss = total_loss / n_batches
    avg_ce_loss = total_ce_loss / n_batches

    logger.info(f'Epoch {epoch} - Average Loss: {avg_loss:.4f}, Average CE Loss: {avg_ce_loss:.4f}')
    if total_graph_loss > 0:
        avg_graph_loss = total_graph_loss / n_batches
        logger.info(f'Epoch {epoch} - Average Graph Loss: {avg_graph_loss:.4f}')

    # Print epoch summary
    print(f'Epoch {epoch} - Average Loss: {avg_loss:.4f}, Average CE Loss: {avg_ce_loss:.4f}', end='')
    if total_graph_loss > 0:
        print(f', Average Graph Loss: {avg_graph_loss:.4f}', end='')
    print()

    finish = time.time()
    time_elapsed = finish - start

    logger.info(f'Epoch {epoch} training completed')
    logger.info(f'Epoch {epoch} training time consumed: {time_elapsed:.2f}s')
    print(f'epoch {epoch} training time consumed: {time_elapsed:.2f}s')


def get_state_list(args):
    """
    Generate state list based on the mode and parameters passed.

    Returns:
        List of 0s and 1s with length num_elements + 1, where 1 indicates active blocks
    """
    # Create state list with all zeros initially (with length num_elements + 1)
    state_list = [0] * (args.num_elements + 1)

    # Always set the last element (output layer) to 1 for output graph calculation
    state_list[args.num_elements] = 1

    # If using specific graph indices
    if args.graph_indices:
        indices = [int(idx) for idx in args.graph_indices.split(',')]
        for idx in indices:
            if idx < 0 or idx >= args.num_elements:
                raise ValueError(
                    f"Invalid index {idx} in graph_indices. Indices must be between 0 and {args.num_elements - 1}.")
            state_list[idx] = 1
        return state_list

    # If using stage-based selection
    if args.stage_mode:
        # Calculate the size of each stage
        num_elements = args.num_elements
        early_size = num_elements // 3
        middle_size = num_elements // 3
        late_size = num_elements - early_size - middle_size

        # Define start and end indices for each stage
        early_range = range(0, early_size)
        middle_range = range(early_size, early_size + middle_size)
        late_range = range(early_size + middle_size, num_elements)

        # Parse the stages and set corresponding elements to 1
        stages = args.stage_mode.split('+')
        for stage in stages:
            if stage == 'early':
                for idx in early_range:
                    state_list[idx] = 1
            elif stage == 'middle':
                for idx in middle_range:
                    state_list[idx] = 1
            elif stage == 'late':
                for idx in late_range:
                    state_list[idx] = 1
            else:
                raise ValueError(f"Invalid stage '{stage}' in stage_mode. Use 'early', 'middle', or 'late'.")

        return state_list

    # Default: use all indices (original behavior)
    for i in range(args.num_elements):
        state_list[i] = 1

    return state_list


def calculate_weight(index, num_elements, method, num_active_graphs=None):
    """
    Calculate weight based on different methods

    Args:
        index: The index position (0-based)
        num_elements: Number of elements in the model
        method: Weighting method (linear, sqrt, squared, equal, arccos, cosine, alpha)
        num_active_graphs: Number of active graphs (used for equal weighting)

    Returns:
        Calculated weight

    Note: For alpha method, this function is not used.
    """
    # For alpha method, return 1.0 directly
    if method == 'alpha':
        return 1.0

    # Convert index to normalized position (0 to 1)
    x = (1 + index) / num_elements

    if method == 'linear':
        return x
    elif method == 'sqrt':
        return math.sqrt(x)
    elif method == 'squared':
        return x ** 2
    elif method == 'equal':
        # Distribute weight evenly among active graphs
        if num_active_graphs and num_active_graphs > 0:
            return 1.0 / num_active_graphs
        return 1.0
    elif method == 'arccos':
        return math.acos(1 - 2 * x) / math.pi
    elif method == 'cosine':
        return (1 + math.cos(math.pi * x)) / 2
    else:
        # Default to linear if invalid method
        logger.warning(f"Invalid weight method '{method}'. Using linear weighting.")
        return x


@torch.no_grad()
def eval_training(epoch=0, tb=True):
    start = time.time()
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for (images, labels) in test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        # Modified forward pass for evaluation - don't calculate any graphs
        outputs, _ = net(images, [0] * (args.num_elements + 1))  # Pass empty state list for evaluation
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    time_elapsed = finish - start

    avg_loss = test_loss / len(test_loader.dataset)
    accuracy = correct.float() / len(test_loader.dataset)

    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
        logger.debug('GPU INFO:\n' + torch.cuda.memory_summary())

    print('Evaluating Network.....')
    print(
        f'Test set: Epoch: {epoch}, Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Time consumed: {time_elapsed:.2f}s')
    print()

    logger.info(f'Evaluation completed for epoch {epoch}')
    logger.info(f'Test set: Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Time consumed: {time_elapsed:.2f}s')

    # Log test metrics
    if tb:  # Keep the tb parameter for compatibility but use it to determine if this is a training run
        logger.info(f'Test/Average loss: {avg_loss:.6f}, epoch: {epoch}')
        logger.info(f'Test/Accuracy: {accuracy:.6f}, epoch: {epoch}')

        # Log current best accuracy if this is better
        global best_acc
        if accuracy > best_acc:
            logger.info(f'New best accuracy: {accuracy:.6f} (previous: {best_acc:.6f})')

    return accuracy


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='cifar100', 
                        choices=['cifar10', 'cifar100', 'tiny_imagenet'],
                        help='dataset to use for training')
    parser.add_argument('-net', type=str, default="mobilenet", help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    # Modified graph loss parameters to match requirements
    parser.add_argument('-graph_loss_weight', type=float, default=1.0, help='weight for graph loss')
    parser.add_argument('-num_elements', type=int, default=6,
                        help='number of elements in the model (state_list length will be num_elements + 1)')
    parser.add_argument('-graph_indices', type=str, default=None,
                        help='comma-separated indices of blocks to calculate graph loss for, e.g., "1,4,5"')
    # Add stage_mode parameter for selecting blocks by stages
    parser.add_argument('-stage_mode', type=str, default=None,
                        help='stages to calculate graph loss for, e.g., "early", "middle", "late", "early+middle", "middle+late", or "early+late"')
    parser.add_argument('-weight_method', type=str, default='linear',
                        choices=['linear', 'sqrt', 'squared', 'equal', 'arccos', 'cosine', 'alpha'],
                        # Added 'alpha' choice
                        help='method for weighting graph loss by layer position')
    parser.add_argument('-use_detach', action='store_true', default=False,
                        help='whether to use detach() on ground_truth_embedding when calculating loss')
    parser.add_argument('-detach_alpha', action='store_true', default=False,  # New parameter for alpha detach
                        help='whether to use detach() on alpha values when using alpha weighting method')
    parser.add_argument('-log', type=str, default='my_logs', help='directory to save log files')
    parser.add_argument('-log_name', type=str, default=None,
                        help='name of the log file (defaults to {net}_{timestamp}.log if not specified)')
    parser.add_argument('-best_checkpoint', type=str, default=None,
                        help='directory to save the best model (defaults to checkpoint_path if not specified)')
    args = parser.parse_args()

    # Setup logging
    if not os.path.exists(args.log):
        os.makedirs(args.log)

    # Create timestamp for log file name if needed
    time_now = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Use user-specified log name if provided, otherwise use default with timestamp
    if args.log_name:
        log_file = os.path.join(args.log, args.log_name)
    else:
        log_file = os.path.join(args.log, f'{args.dataset}_{args.net}_{time_now}.log')

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Create logger
    logger = logging.getLogger(f"{args.dataset}_{args.net}")
    logger.info(f"{'=' * 20} Training started at {time_now} {'=' * 20}")
    logger.info(f"Command line arguments: {args}")

    # Check for conflicting arguments
    if args.graph_indices and args.stage_mode:
        logger.warning("Both -graph_indices and -stage_mode provided. Using -graph_indices and ignoring -stage_mode.")

    # Validate dataset and get configuration
    try:
        dataset_config = settings.get_dataset_config(args.dataset)
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Number of classes: {dataset_config['num_classes']}")
    except ValueError as e:
        logger.error(f"Dataset configuration error: {e}")
        sys.exit(1)

    net = get_network(args)
    logger.info(f"Network: {args.net}")

    # Get dataset loaders
    training_loader, test_loader = get_dataset_loaders(args.dataset, args)

    logger.info(f"Training samples: {len(training_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Print dataset directory structure information
    if args.dataset == 'cifar10':
        logger.info("Expected data structure: ./data/ (CIFAR-10 will be downloaded automatically)")
    elif args.dataset == 'cifar100':
        logger.info("Expected data structure: ./data/ (CIFAR-100 will be downloaded automatically)")
    elif args.dataset == 'tiny_imagenet':
        logger.info("Expected data structure: ./data/tiny/ (Please download Tiny ImageNet dataset)")
        logger.info("  - ./data/tiny/train/")
        logger.info("  - ./data/tiny/val/")

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES,
                                                     gamma=0.2)  # learning rate decay
    iter_per_epoch = len(training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    logger.info(f"Optimizer: SGD (lr={args.lr}, momentum=0.9, weight_decay=5e-4)")
    logger.info(f"Learning rate milestones: {settings.MILESTONES}")
    logger.info(f"Model elements: {args.num_elements} (state_list length: {args.num_elements + 1})")
    logger.info(f"Weight method: {args.weight_method}")

    # Log the appropriate mode information
    if args.graph_indices:
        logger.info(f"Using graph loss with indices: {args.graph_indices}")
    elif args.stage_mode:
        # Calculate stage divisions for logging
        early_size = args.num_elements // 3
        middle_size = args.num_elements // 3
        late_size = args.num_elements - early_size - middle_size
        early_range = f"0-{early_size - 1}"
        middle_range = f"{early_size}-{early_size + middle_size - 1}"
        late_range = f"{early_size + middle_size}-{args.num_elements - 1}"

        logger.info(f"Using stage mode: {args.stage_mode}")
        logger.info(f"Stage divisions: early [{early_range}], middle [{middle_range}], late [{late_range}]")
    else:
        logger.info(f"Using graph loss with all indices (0 to {args.num_elements - 1})")

    logger.info(f"Graph loss weight: {args.graph_loss_weight}")
    logger.info(f"Using detach on ground_truth_embedding: {args.use_detach}")
    if args.weight_method == 'alpha':
        logger.info(f"Using detach on alpha values: {args.detach_alpha}")  # Log alpha detach setting

    # Initialize or restore best_acc and tracking variables
    best_acc = 0.0
    best_epoch = 0

    if args.resume:
        # If best_checkpoint is specified, try to load from there first
        if args.best_checkpoint and os.path.exists(os.path.join(args.best_checkpoint, f'{args.net}-best.pth')):
            weights_path = os.path.join(args.best_checkpoint, f'{args.net}-best.pth')
            logger.info(f'Found best model at specified best_checkpoint: {weights_path}')
            logger.info('Loading best model to resume training...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            logger.info(f'Previous best acc: {best_acc:.6f}')
            resume_epoch = 0  # We don't know which epoch this was from, so start from the next epoch
        else:
            # Fall back to the traditional resume logic
            recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net),
                                               fmt=settings.DATE_FORMAT)
            if not recent_folder:
                raise Exception('no recent folder were found')

            checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")

            best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
            if best_weights:
                weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
                logger.info(f'Found best acc weights file: {weights_path}')
                logger.info('Loading best training file to test acc...')
                net.load_state_dict(torch.load(weights_path))
                best_acc = eval_training(tb=False)
                logger.info(f'Previous best acc: {best_acc:.6f}')

            # Only load the most recent weights if we didn't find a best weights file
            if not best_weights:
                recent_weights_file = most_recent_weights(
                    os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
                if not recent_weights_file:
                    raise Exception('no recent weights file were found')
                weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
                logger.info(f'Loading weights file {weights_path} to resume training')
                net.load_state_dict(torch.load(weights_path))

            resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
            logger.info(f'Resume from epoch {resume_epoch}')
    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, time_now)
        logger.info(f"Checkpoint path for logs: {checkpoint_path}")

    start_time = time.time()
    logger.info(f"Training start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup best model checkpoint path
    if args.best_checkpoint:
        if not os.path.exists(args.best_checkpoint):
            os.makedirs(args.best_checkpoint)
        best_model_path = os.path.join(args.best_checkpoint, f'{args.dataset}_{args.net}-best.pth')
    else:
        best_model_path = os.path.join(checkpoint_path, f'{args.dataset}_{args.net}-best.pth')

    logger.info(f"Best model will be saved to: {best_model_path}")

    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)
        acc = eval_training(epoch)

        # Save best performance model immediately when improvement is detected
        if best_acc < acc:
            prev_best = best_acc
            best_acc = acc
            best_epoch = epoch

            # Save the model
            logger.info(f'Saving best weights file to {best_model_path}')
            print(f'Saving best weights file to {best_model_path}')
            torch.save(net.state_dict(), best_model_path)

            # Log the improvement
            logger.info(f'Best accuracy updated: {prev_best:.6f} -> {best_acc:.6f} at epoch {best_epoch}')

    end_time = time.time()
    training_time = end_time - start_time
    logger.info(f"Training end time: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total training time: {training_time:.2f} seconds ({training_time / 3600:.2f} hours)")
    logger.info(f"Final best accuracy: {best_acc:.6f} achieved at epoch {best_epoch}")

    # Output detailed best performance record
    if args.graph_indices:
        mode_info = f"graph indices: {args.graph_indices}"
    elif args.stage_mode:
        mode_info = f"stage mode: {args.stage_mode}"
    else:
        mode_info = "all indices"

    logger.info(
        f"Best configuration: dataset: {args.dataset}, num_elements: {args.num_elements}, {mode_info}, weight method: {args.weight_method}")
    logger.info(f"use_detach: {args.use_detach}, graph loss weight: {args.graph_loss_weight}")
    if args.weight_method == 'alpha':
        logger.info(f"detach_alpha: {args.detach_alpha}")  # Log alpha detach setting in final summary

    # Log where the best model was saved
    if args.best_checkpoint:
        logger.info(f"Best model saved to: {os.path.join(args.best_checkpoint, f'{args.dataset}_{args.net}-best.pth')}")
    else:
        logger.info(f"Best model saved to: {os.path.join(checkpoint_path, f'{args.dataset}_{args.net}-best.pth')}")

    logger.info(f"{'=' * 20} Training completed {'=' * 20}")

    print(f"Training completed. Total time: {training_time / 3600:.2f} hours")
    print(f"Final best accuracy: {best_acc:.6f} achieved at epoch {best_epoch}")
    print(
        f"Best configuration: dataset: {args.dataset}, num_elements: {args.num_elements}, {mode_info}, weight method: {args.weight_method}")

    # Print best model path
    if args.best_checkpoint:
        print(f"Best model saved to: {os.path.join(args.best_checkpoint, f'{args.dataset}_{args.net}-best.pth')}")
    else:
        print(f"Best model saved to: {os.path.join(checkpoint_path, f'{args.dataset}_{args.net}-best.pth')}")

    print(f"Log file saved to: {log_file}")