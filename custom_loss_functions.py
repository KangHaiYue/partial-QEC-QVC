import torch
import torch.nn as nn
from torch import Tensor

class SoftMaxSmoothedPQCLoss(nn.Module):
    def __init__(self):
        super().__init__()        
        self.softmax = nn.Softmax(dim=1)
        
    
    def forward(self, values: Tensor, targets: Tensor) -> float:
        """
        values: tensor of shape (batch_size, num_classes)
        targets: tensor of shape (batch_size, num_classes) with one-hot encoding
        """
        
        minibatch_size = values.shape[0]
        if minibatch_size != targets.shape[0]:
            raise Exception('batch size in loss_values does not match target')
        # find the targets' inverse CDF
        probs = self.softmax(values)
        #standard_inv_CDFs = torch.distributions.normal.Normal(0,1).icdf(probs)
        #target_inv_CDFs = torch.sum(standard_inv_CDFs*targets, dim=1)
        target_probs = torch.sum(probs*targets, dim=1)
        
        
        # find the largest non-target inverse CDF (the prob of the target is forced to be at most 1e-50, 
        # thus guaranteeing that the largest non-target inverse CDF is always larger than the target inverse CDF)
        non_targets_probs = probs*(1-targets)#*(1-1e-50))
        #non_targets_CDFs = torch.distributions.normal.Normal(0,1).icdf(non_targets_probs)
        #largest_non_target_inv_CDFs = torch.max(non_targets_CDFs, dim=1).values
        largest_non_target_probs = torch.max(non_targets_probs, dim=1).values
        
        
        #return -torch.sum(target_inv_CDFs - largest_non_target_inv_CDFs)
        return torch.sum(-1 + torch.log(largest_non_target_probs) - torch.log(target_probs))/minibatch_size
        
        #cross_entropy = torch.sum(torch.Tensor.log(nn.functional.softmax(values,dim=1))*targets)/len(values)
        #return cross_entropy - mean_variance


class ExpvalSmoothedPQCLoss(nn.Module):
    def __init__(self):
        super().__init__()        
            
    def forward(self, values: Tensor, targets: Tensor) -> float:
        """
        values: tensor of shape (batch_size, num_classes)
        targets: tensor of shape (batch_size, num_classes) with one-hot encoding
        """
        
        minibatch_size = values.shape[0]
        if minibatch_size != targets.shape[0]:
            raise Exception('batch size in loss_values does not match target')

        target_expectation_O = torch.sum(values*targets, dim=1)
        
        
        # find the largest non-target inverse CDF (the prob of the target is forced to be at most 1e-50, 
        # thus guaranteeing that the largest non-target inverse CDF is always larger than the target inverse CDF)
        non_target_expectation_O = (values + 1e10) * (1 - targets) - 1e10
        #non_targets_CDFs = torch.distributions.normal.Normal(0,1).icdf(non_targets_probs)
        #largest_non_target_inv_CDFs = torch.max(non_targets_CDFs, dim=1).values
        largest_non_target_expectation_O = torch.max(non_target_expectation_O, dim=1).values
        
        
        #return -torch.sum(target_inv_CDFs - largest_non_target_inv_CDFs)
        return torch.sum(-2 + largest_non_target_expectation_O - target_expectation_O)/minibatch_size
        
        #cross_entropy = torch.sum(torch.Tensor.log(nn.functional.softmax(values,dim=1))*targets)/len(values)
        #return cross_entropy - mean_variance
        
class NormalSmoothedPQCLoss(nn.Module):
    def __init__(self):
        super().__init__()        
            
    def forward(self, values: Tensor, targets: Tensor) -> float:
        """
        values: tensor of shape (batch_size, num_classes)
        targets: tensor of shape (batch_size, num_classes) with one-hot encoding
        """
        
        minibatch_size = values.shape[0]
        if minibatch_size != targets.shape[0]:
            raise Exception('batch size in loss_values does not match target')
        
        
        targets_indexs = torch.argmax(targets, dim=1)
        values_targets_removed = torch.stack([torch.cat([values[i, :targets_indexs[i]], values[i, targets_indexs[i]+1:]]) 
                                              for i in range(values.shape[0])])
        
        
        values_mean = torch.mean(values_targets_removed, dim=1)
        values_variance = torch.var(values_targets_removed, dim=1)
        
        target_expectation_O = torch.sum(values*targets, dim=1)
        
        return -torch.sum(torch.distributions.normal.Normal(values_mean, values_variance).cdf(target_expectation_O))/minibatch_size