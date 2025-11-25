import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from .models.mlp import MLP
from .defenses import adversarial_training_step

class TorchTrainer:
    def __init__(self, in_dim, hidden=(512,256,128), dropout=0.2, device=None):
        self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = MLP(in_dim, hidden, dropout).to(self.device)

    def loaders(self, X_tr, y_tr, X_va, y_va, batch=4096):
        tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        va = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
        self.tr_loader = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True)
        self.va_loader = DataLoader(va, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)

    def fit(self, epochs=10, lr=1e-3, wd=1e-5, amp=True,
            adv_train=False, attack='pgd', eps=0.1, pgd_steps=5, pgd_step_size=0.02, adv_frac=0.5,
            lo=None, hi=None, ckpt=None):
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        best = 0.0
        for e in range(1, epochs+1):
            self.model.train()
            p = tqdm(self.tr_loader, desc=f"Epoch {e}/{epochs}")
            for xb, yb in p:
                xb, yb = xb.to(self.device), yb.to(self.device)
                if adv_train:
                    xb, yb = adversarial_training_step(self.model, xb, yb, attack=attack, eps=eps,
                                                       step_size=pgd_step_size, steps=pgd_steps,
                                                       lo=lo, hi=hi, frac=adv_frac)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp):
                    loss = F.binary_cross_entropy_with_logits(self.model(xb), yb)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                p.set_postfix({'loss': f'{loss.item():.4f}'})
            acc, auc = self.evaluate()
            print(f"Val acc={acc:.4f} auc={auc:.4f}")
            if acc > best and ckpt:
                os.makedirs(os.path.dirname(ckpt), exist_ok=True)
                torch.save({'model_state': self.model.state_dict(), 'val_acc': acc, 'val_auc': auc}, ckpt)
                best = acc; print(f"Saved -> {ckpt}")

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        ys, ps, probs = [], [], []
        for xb, yb in self.va_loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            lg = self.model(xb)
            p = (torch.sigmoid(lg) > 0.5).float()
            pr = torch.sigmoid(lg)
            ys.append(yb.cpu().numpy()); ps.append(p.cpu().numpy()); probs.append(pr.cpu().numpy())
        from sklearn.metrics import accuracy_score, roc_auc_score
        import numpy as np
        y = np.concatenate(ys); yhat = np.concatenate(ps); pr = np.concatenate(probs)
        acc = accuracy_score(y, yhat)
        try: auc = roc_auc_score(y, pr)
        except: auc = float('nan')
        return acc, auc

    def eval_under_attack(self, attack='pgd', eps=0.1, steps=5, step_size=0.02, lo=None, hi=None):
        from .attacks.fgsm_pgd import fgsm, pgd
        self.model.eval()
        ys, ps, probs = [], [], []
        for xb, yb in self.va_loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            if attack == 'fgsm': xadv = fgsm(self.model, xb, yb, eps, lo, hi)
            elif attack == 'pgd': xadv = pgd(self.model, xb, yb, eps, step_size, steps, lo, hi)
            else: raise ValueError('attack must be fgsm or pgd')
            lg = self.model(xadv)
            p = (torch.sigmoid(lg) > 0.5).float(); pr = torch.sigmoid(lg)
            ys.append(yb.cpu().numpy()); ps.append(p.cpu().numpy()); probs.append(pr.cpu().numpy())
        from sklearn.metrics import accuracy_score, roc_auc_score
        import numpy as np
        y = np.concatenate(ys); yhat = np.concatenate(ps); pr = np.concatenate(probs)
        acc = accuracy_score(y, yhat)
        try: auc = roc_auc_score(y, pr)
        except: auc = float('nan')
        return acc, auc
