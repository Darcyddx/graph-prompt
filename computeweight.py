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


def calculate_weight(index, num_elements, method, num_active_graphs=None):
    if method == 'adaptive':
        return 1.0

    x = (1 + index) / num_elements

    if method == 'linear':
        return x
    elif method == 'sqrt':
        return math.sqrt(x)
    elif method == 'squared':
        return x ** 2
    elif method == 'equal':
        if num_active_graphs and num_active_graphs > 0:
            return 1.0 / num_active_graphs
        return 1.0
    elif method == 'arccos':
        return math.acos(1 - 2 * x) / math.pi
    elif method == 'cosine':
        return (1 + math.cos(math.pi * x)) / 2
    else:
        logger.warning(f"Invalid weight method '{method}'. Using linear weighting.")
        return x