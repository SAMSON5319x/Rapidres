import torch
from .attacks.fgsm_pgd import fgsm, pgd

@torch.no_grad()
def _mix(clean_x, clean_y, adv_x, adv_y):
    return torch.cat([adv_x, clean_x],0), torch.cat([adv_y, clean_y],0)

def adversarial_training_step(model, xb, yb, *, attack, eps, step_size, steps, lo, hi, frac=0.5):
    b = xb.size(0); k = int(b*frac)
    if k == 0:
        return xb, yb
    x_sel, y_sel = xb[:k], yb[:k]
    if attack == 'fgsm':
        x_adv = fgsm(model, x_sel, y_sel, eps, lo, hi)
    elif attack == 'pgd':
        x_adv = pgd(model, x_sel, y_sel, eps, step_size, steps, lo, hi)
    else:
        raise ValueError("attack must be fgsm or pgd")
    return _mix(xb[k:], yb[k:], x_adv, y_sel)