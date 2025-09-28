import os
import sys
import re
import datetime
import numpy
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class TinyImageNetDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        # Load data
        self.image_paths = []
        self.labels = []

        if split == 'train':
            split_path = os.path.join(root, 'train')
            # Get all class directories
            class_dirs = os.listdir(split_path)
            class_dirs.sort()  # Ensure consistent ordering

            # Create class to index mapping
            self.class_to_idx = {class_dir: idx for idx, class_dir in enumerate(class_dirs)}

            # Load all images
            for class_dir in class_dirs:
                class_path = os.path.join(split_path, class_dir, 'images')
                if not os.path.exists(class_path):
                    continue

                for img_name in os.listdir(class_path):
                    if img_name.endswith('.JPEG'):
                        img_path = os.path.join(class_path, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[class_dir])

        elif split == 'val':
            split_path = os.path.join(root, 'val')

            # Read validation annotations
            val_annotations = os.path.join(split_path, 'val_annotations.txt')
            class_names = []

            # First pass to get all unique class names
            with open(val_annotations, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    class_name = parts[1]
                    if class_name not in class_names:
                        class_names.append(class_name)

            class_names.sort()  # Ensure consistent ordering
            self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

            # Second pass to load images
            with open(val_annotations, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    img_name = parts[0]
                    class_name = parts[1]

                    img_path = os.path.join(split_path, 'images', img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

        print(f"{split} dataset loaded: {len(self.image_paths)} images, {len(self.class_to_idx)} classes")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_network(args):
    """ return given network
    """

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
    elif args.net == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif args.net == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif args.net == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif args.net == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()
    # Vision Transformers and modern architectures (from Tiny ImageNet version)
    elif args.net == 'vit_b_16':
        from models.vit import vit_b_16
        net = vit_b_16()
    elif args.net == 'vit_b_32':
        from models.vit import vit_b_32
        net = vit_b_32()
    elif args.net == 'swin_b':
        from models.swinformer import swin_b
        net = swin_b()
    elif args.net == 'swin_s':
        from models.swinformer import swin_s
        net = swin_s()
    elif args.net == 'swin_t':
        from models.swinformer import swin_t
        net = swin_t()
    elif args.net == 'mobilevit_s':
        from models.mobilevit import mobilevit_s
        net = mobilevit_s()
    elif args.net == 'mobilevit_xs':
        from models.mobilevit import mobilevit_xs
        net = mobilevit_xs()
    elif args.net == 'mobilevit_xxs':
        from models.mobilevit import mobilevit_xxs
        net = mobilevit_xxs()
    elif args.net == 'ceit_t':
        from models.ceit import ceit_t
        net = ceit_t()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net


# CIFAR-10 functions
def get_cifar10_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return cifar10 training dataloader
    Args:
        mean: mean of cifar10 training dataset
        std: std of cifar10 training dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    cifar10_training = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    cifar10_training_loader = DataLoader(
        cifar10_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_training_loader


def get_cifar10_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=False):
    """ return cifar10 test dataloader
    Args:
        mean: mean of cifar10 test dataset
        std: std of cifar10 test dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar10_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    cifar10_test_loader = DataLoader(
        cifar10_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_test_loader


# CIFAR-100 functions
def get_cifar100_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return cifar100 training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader


def get_cifar100_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=False):
    """ return cifar100 test dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader


# Tiny ImageNet functions
def get_tiny_imagenet_training_dataloader(mean, std, batch_size=16, num_workers=8, shuffle=True, data_root='./data', img_size=64):
    """ return tiny imagenet training dataloader
    Args:
        mean: mean of tiny imagenet training dataset
        std: std of tiny imagenet training dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
        data_root: path to tiny imagenet dataset
        img_size: image size for preprocessing (64 for standard models, 224 for ViT)
    Returns: train_data_loader:torch dataloader object
    """

    if img_size == 224:
        # ViT preprocessing: resize and use larger crop
        transform_train = transforms.Compose([
            transforms.Resize(256),  # Resize to 256 first
            transforms.RandomCrop(224, padding=16),  # 224x224 crop with 16 pixel padding
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        # Standard preprocessing for 64x64
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=8),  # 64x64 images with 8 pixel padding
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    tiny_imagenet_training = TinyImageNetDataset(root=data_root, split='train', transform=transform_train)
    tiny_imagenet_training_loader = DataLoader(
        tiny_imagenet_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=True)

    return tiny_imagenet_training_loader


def get_tiny_imagenet_test_dataloader(mean, std, batch_size=16, num_workers=8, shuffle=False, data_root='./data', img_size=64):
    """ return tiny imagenet test dataloader
    Args:
        mean: mean of tiny imagenet test dataset
        std: std of tiny imagenet test dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
        data_root: path to tiny imagenet dataset
        img_size: image size for preprocessing (64 for standard models, 224 for ViT)
    Returns: tiny_imagenet_test_loader:torch dataloader object
    """

    if img_size == 224:
        # ViT preprocessing: resize and center crop
        transform_test = transforms.Compose([
            transforms.Resize(256),  # Resize to 256 first
            transforms.CenterCrop(224),  # Center crop to 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        # Standard preprocessing for 64x64
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    tiny_imagenet_test = TinyImageNetDataset(root=data_root, split='val', transform=transform_test)
    tiny_imagenet_test_loader = DataLoader(
        tiny_imagenet_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=True)

    return tiny_imagenet_test_loader


# Utility functions for computing dataset statistics
def compute_cifar_mean_std(cifar_dataset):
    """compute the mean and std of cifar dataset
    Args:
        cifar_training_dataset or cifar_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar_dataset[i][1][:, :, 0] for i in range(len(cifar_dataset))])
    data_g = numpy.dstack([cifar_dataset[i][1][:, :, 1] for i in range(len(cifar_dataset))])
    data_b = numpy.dstack([cifar_dataset[i][1][:, :, 2] for i in range(len(cifar_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std


def compute_tiny_imagenet_mean_std(dataset):
    """compute the mean and std of tiny imagenet dataset
    Args:
        dataset: dataset object

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    means = []
    stds = []

    for i in range(len(dataset)):
        img, _ = dataset[i]
        # Convert PIL Image to numpy array
        img_array = numpy.array(img)

        # Compute mean and std for each channel
        for c in range(3):
            channel_data = img_array[:, :, c].flatten()
            means.append(numpy.mean(channel_data))
            stds.append(numpy.std(channel_data))

    # Average across all images
    final_means = []
    final_stds = []

    for c in range(3):
        channel_means = means[c::3]
        channel_stds = stds[c::3]
        final_means.append(numpy.mean(channel_means) / 255.0)
        final_stds.append(numpy.mean(channel_stds) / 255.0)

    return tuple(final_means), tuple(final_stds)


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    folders = os.listdir(net_weights)

    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]


def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]


def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch


def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]