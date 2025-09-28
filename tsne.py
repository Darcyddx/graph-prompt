import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# TODO: import the model you want to apply t-SNE
from models.shufflenet import shufflenet
from models.mobilenet import mobilenet


# Define colors for different classes
COLORS = [
    '#0047AB',  # Cobalt blue (airplane - non-animal)
    '#8B0000',  # Dark red (car - non-animal)
    '#FF3300',  # Bright orange-red (bird - animal)
    '#FFCC00',  # Bright golden yellow (cat - animal)
    '#33CC33',  # Bright green (deer - animal)
    '#FF66FF',  # Bright pink (dog - animal)
    '#00FFCC',  # Bright cyan (frog - animal)
    '#FF9900',  # Bright orange (horse - animal)
    '#800080',  # Purple (ship - non-animal)
    '#006400'  # Dark green (truck - non-animal)
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=300, shuffle=False, num_workers=2
)


def extract_features(model_path):
    model = shufflenet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    feature_extractor = FeatureExtractor(model).to(device)

    features = []
    labels = []

    with torch.no_grad():
        for data in testloader:
            images, targets = data
            images = images.to(device)
            outputs = feature_extractor(images)
            features.append(outputs.cpu().numpy())
            labels.append(targets.numpy())

    features = np.vstack(features)
    labels = np.concatenate(labels)

    return features, labels


def visualize_tsne(features, labels, title, save_path):
    print(f"Applying t-SNE to {features.shape[0]} samples with {features.shape[1]} features...")
    tsne = TSNE(n_components=2, random_state=2025, perplexity=30)
    features_tsne = tsne.fit_transform(features)

    plt.figure(figsize=(10, 10))
    for i in range(10):
        idx = labels == i
        plt.scatter(
            features_tsne[idx, 0],
            features_tsne[idx, 1],
            c=COLORS[i],
            s=10,
            alpha=0.7
        )

    plt.xticks([])
    plt.yticks([])

    plt.savefig(save_path, dpi=50, bbox_inches='tight')
    print(f"Visualization saved at {save_path}")


def main():
    # TODO: Modify the path to your model checkpoint
    model_path = 'path/to/your/model/checkpoint.pth'
    
    print("Extracting features from model...")
    features, labels = extract_features(model_path)

    print("Visualizing t-SNE...")
    visualize_tsne(features, labels, "t-SNE Visualization", "tsne_result.pdf")


if __name__ == "__main__":
    main()