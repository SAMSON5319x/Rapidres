import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from .fgsm_pgd import fgsm, pgd

class SurrogateTransfer:
    def __init__(self, surrogate_model, device=None):
        self.s = surrogate_model
        self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.s.to(self.device).eval()

    @torch.no_grad()
    def _acc(self, target, X, y):
        yhat = target.predict(X)
        return accuracy_score(y, yhat)

    def craft_and_transfer(self, target_model, X, y, attack='fgsm', eps=0.1, steps=5, step_size=0.02):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        ld = DataLoader(ds, batch_size=8192, shuffle=False)
        adv_list = []
        for xb, yb in ld:
            xb, yb = xb.to(self.device), yb.to(self.device)
            lo = xb.min(dim=0).values; hi = xb.max(dim=0).values
            if attack == 'fgsm': xadv = fgsm(self.s, xb, yb, eps, lo, hi)
            else: xadv = pgd(self.s, xb, yb, eps, step_size, steps, lo, hi)
            adv_list.append(xadv.cpu().numpy())
        X_adv = np.concatenate(adv_list)
        clean_acc = self._acc(target_model, X, y)
        adv_acc = self._acc(target_model, X_adv, y)
        transfer_success = float(np.mean(target_model.predict(X) != target_model.predict(X_adv)))
        return X_adv, clean_acc, adv_acc, transfer_success
