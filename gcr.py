import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class RelationshipLayer(nn.Module):
    def __init__(self, similarity='cos', gamma=None):
        super(RelationshipLayer, self).__init__()
        self.similarity = similarity
        self.gamma = gamma
        self.relu = nn.ReLU()

    def forward(self, x, state):
        if not state:
            return None  

        if len(x.shape) != 2:
            x = x.view(x.shape[0], -1) if x.is_contiguous() else x.reshape(x.shape[0], -1)

        B = x.size(0)  

        if self.similarity == 'cos':
            x_norm = F.normalize(x, p=2, dim=1)
            sim_matrix = self.relu(x_norm @ x_norm.T)
        elif self.similarity == 'rbf':
            if self.gamma is None:
                self.gamma = 1.0 / x.size(1)  
            dist_matrix = torch.cdist(x, x, p=2) ** 2 
            sim_matrix = torch.exp(-self.gamma * torch.clamp(dist_matrix, max=100))
        else:
            raise ValueError("unsupported similarity type. choose 'cosine' or 'rbf'.")

        upper_tri_indices = torch.triu_indices(B, B, offset=1)
        upper_tri_matrix = sim_matrix[upper_tri_indices[0], upper_tri_indices[1]]

        return upper_tri_matrix