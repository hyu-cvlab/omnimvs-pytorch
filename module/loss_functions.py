# module.loss_functions.py
#
# Author: Changhee Won (changhee.1.won@gmail.com)
#
#

from module.basic import *

def entropy_boundary_loss(x, k, a, reduction='mean'):
    x = torch.relu(a*(x-np.log(k)))
    if reduction != 'mean':
        return torch.sum(x)
    else:
        return torch.mean(x)

def ent_exploss(x, k, a):
    inv_k = 1.0/k
    return torch.relu(a*inv_k*(torch.exp(x)-k))

def ent_huber(x, sigma):
    return torch.where(x<sigma, 0.5*x**2, sigma*(x-0.5*sigma))

def ent_huberk(x, sigma, k):
    inv_k = 1.0/k
    sigma_k = sigma**(k-1)
    sigma_k2 = (k-1.0)/k*sigma
    return torch.where(x<sigma, inv_k*x**k, sigma_k*(x-sigma_k2))