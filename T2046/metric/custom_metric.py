import torch

def onehot_f1_score(outputs, targets):
    tp = (targets*outputs).sum().item()
    fp = ((1-targets)*outputs).sum().item()
    fn = (targets*(1-outputs)).sum().item()
    
    epsilon = 1e-7

    precision = tp / (tp+fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = (2*precision*recall)/(recall + precision + epsilon)
    
    return f1
                
        