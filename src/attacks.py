import numpy
import torch
import torch.functional as F


# TODO check if I need to add to device and can x and y be vectors or should this be done iteratively
# Fast-Gradient Sign Method for computing adversial attacks
def FGSM(model, x, y, epsilon):
    # gradient with respect to input only to save compute
    x.requires_grad = True
    for param in model.parameters():
        param.requires_grad = False

    output = model(x)
    loss = F.nll_loss(output, y)
    loss.backward()

    # set it back to prevent bugs
    for param in model.parameters():
        param.requires_grad = True

    # create adversial examples
    x_adv = x + epsilon * x.grad.sign()
    return x_adv


# Saliency attack from Papernot et al.
def saliency_map():
    pass
