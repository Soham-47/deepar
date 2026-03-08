import torch

def Gausian_NLL(mu, sigma, y_true):
    return (
        (torch.log(2 * torch.pi * sigma**2) / 2) + 
        ((y_true - mu)**2 / (2 * sigma**2))
    ).mean()
