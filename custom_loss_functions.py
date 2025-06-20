import torch
import torch.nn as nn
from torch.types import Tensor

class SmoothedPQCLoss(nn.Module):
    def __init__(self):
        super().__init__()        
        self.softmax = nn.Softmax(dim=1)
        
    
    def forward(self, values: Tensor, targets: Tensor) -> float:
        """
        values: tensor of shape (batch_size, num_classes)
        targets: tensor of shape (batch_size, num_classes) with one-hot encoding
        """
        
        batch_size = values.shape[0]
        if batch_size != targets.shape[0]:
            raise Exception('batch size in loss_values does not match target')
        # find the targets' inverse CDF
        probs = self.softmax(values)
        #standard_inv_CDFs = torch.distributions.normal.Normal(0,1).icdf(probs)
        #target_inv_CDFs = torch.sum(standard_inv_CDFs*targets, dim=1)
        target_probs = torch.sum(probs*targets, dim=1)
        
        
        # find the largest non-target inverse CDF (the prob of the target is forced to be at most 1e-50, 
        # thus guaranteeing that the largest non-target inverse CDF is always larger than the target inverse CDF)
        non_targets_probs = probs*(1-targets*(1-1e-50))
        #non_targets_CDFs = torch.distributions.normal.Normal(0,1).icdf(non_targets_probs)
        #largest_non_target_inv_CDFs = torch.max(non_targets_CDFs, dim=1).values
        largest_non_target_probs = torch.max(non_targets_probs, dim=1).values
        
        
        #return -torch.sum(target_inv_CDFs - largest_non_target_inv_CDFs)
        return torch.sum(largest_non_target_probs - target_probs)/batch_size
        
        #cross_entropy = torch.sum(torch.Tensor.log(nn.functional.softmax(values,dim=1))*targets)/len(values)
        #return cross_entropy - mean_variance