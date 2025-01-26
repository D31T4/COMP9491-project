import torch
import torch.nn.functional as F

def classification_accuracy(
    predicted: torch.IntTensor, 
    ground_truth: torch.IntTensor
) -> torch.FloatTensor:
    '''
    signal prediction accuracy
    '''
    return (predicted == ground_truth).float().mean()

def displacement_error(
    predicted: torch.FloatTensor, 
    ground_truth: torch.FloatTensor
) -> torch.FloatTensor:
    '''
    displacement error
    '''
    return F.pairwise_distance(predicted, ground_truth)
