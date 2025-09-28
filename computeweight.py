import torch

def compute_adaptive(L_graph):
    """
    Efficiently compute all alpha values for the graph layers.

    Args:
    - L_graph (torch.Tensor): A tensor of graph losses for each layer of size (L,).

    Returns:
    - torch.Tensor: A tensor of alpha values for each layer, size (L,).
    """
    if not isinstance(L_graph, torch.Tensor):
        L_graph = torch.tensor(L_graph)

    # Compute the exponentials and normalize in one go
    alpha_values = torch.softmax(-L_graph, dim=0)
    
    return alpha_values

