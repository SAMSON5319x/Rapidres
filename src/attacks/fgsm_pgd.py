import torch
import torch.nn.functional as F

def clamp_pf(x, lo, hi):
    return torch.max(torch.min(x, hi), lo)

def fgsm(model, x, y, eps, lo, hi):
    x_adv = x.detach().clone().requires_grad_(True)
    logits = model(x_adv)
    loss = F.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    x_adv = x_adv + eps * x_adv.grad.sign()
    return clamp_pf(x_adv.detach(), lo, hi)

def pgd(model, x, y, eps, step_size, steps, lo, hi, random_start=True):
    x0 = x.detach()
    if random_start:
        x_adv = x0 + (2*torch.rand_like(x0)-1)*eps
        x_adv = clamp_pf(x_adv, lo, hi)
    else:
        x_adv = x0.clone()
    x_adv.requires_grad_(True)
    for _ in range(steps):
        loss = F.binary_cross_entropy_with_logits(model(x_adv), y)
        loss.backward()
        with torch.no_grad():
            x_adv += step_size * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x0+eps), x0-eps)
            x_adv = clamp_pf(x_adv, lo, hi)
        x_adv.grad.zero_()
    return x_adv.detach()
