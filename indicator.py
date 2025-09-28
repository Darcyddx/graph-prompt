import torch

def compute_similarity_indicator(labels):
    """
    Computes a similarity matrix for the batch, where the value is 1 if the samples i and j belong to the same class,
    and 0 if they belong to different classes. The diagonal will be 1 (each sample is similar to itself).
    
    Args:
    - labels (torch.Tensor): A tensor of shape (B,) containing class labels for each sample in the batch.
    
    Returns:
    - torch.Tensor: A BxB similarity matrix where M[i, j] = 1 if samples i and j have the same class label, else 0.
    """
    B = labels.size(0)  # Batch size
    
    # Create a similarity matrix based on class labels
    similarity_indicator = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    
    upper_tri_indices = torch.triu_indices(B, B, offset=1)
    upper_tri_indicator = similarity_indicator[upper_tri_indices[0], upper_tri_indices[1]]

    
    return upper_tri_indicator
