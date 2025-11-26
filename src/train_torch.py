import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from .models.mlp import MLP
from .defenses import adversarial_training_step

# ---------------------------------------------------------
# AMP helpers (PyTorch 1.x / 2.x)
# ---------------------------------------------------------
if hasattr(torch, "amp"):   # PyTorch ≥ 2.0
    AMP_SCALER = torch.cuda.amp.GradScaler  # still the recommended scaler
    def autocast(enabled=True):
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
else:                       # PyTorch ≤ 1.13
    AMP_SCALER = torch.cuda.amp.GradScaler
    def autocast(enabled=True):
        return torch.cuda.amp.autocast(enabled=enabled)


class TorchTrainer:
    def __init__(self, in_dim, hidden=(512,256,128), dropout=0.2, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MLP(in_dim, hidden, dropout).to(self.device)

    def loaders(self, X_tr, y_tr, X_va, y_va, batch=4096, num_workers=0):
        """Create DataLoaders (num_workers=0 for Windows compatibility)."""
        # Ensure numpy arrays (handles pandas Series/DataFrame too)
        X_tr = np.asarray(X_tr, dtype=np.float32)
        X_va = np.asarray(X_va, dtype=np.float32)
        y_tr = np.asarray(y_tr, dtype=np.float32).reshape(-1)
        y_va = np.asarray(y_va, dtype=np.float32).reshape(-1)

        tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        va = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))

        self.tr_loader = DataLoader(tr, batch_size=batch, shuffle=True,
                                    num_workers=num_workers, pin_memory=True)
        self.va_loader = DataLoader(va, batch_size=batch, shuffle=False,
                                    num_workers=num_workers, pin_memory=True)

    def fit(self, epochs=10, lr=1e-3, wd=1e-5, amp=True,
        adv_train=False, attack='pgd', eps=0.1,
        pgd_steps=5, pgd_step_size=0.02, adv_frac=0.5,
        lo=None, hi=None, ckpt=None,
        pos_weight_val=None, label_smooth=0.0, grad_clip=1.0,
        curriculum=True):
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        scaler = torch.amp.GradScaler('cuda', enabled=amp)
        best = 0.0
        pos_weight = None
        if pos_weight_val is not None:
            pos_weight = torch.tensor([pos_weight_val], device=self.device)

        for e in range(1, epochs + 1):
            self.model.train()
            # PGD curriculum: ramp eps and steps during training
            cur_eps = eps * (e / epochs) if (adv_train and curriculum) else eps
            cur_steps = max(3, int(pgd_steps * (e / epochs))) if (adv_train and curriculum) else pgd_steps

            pbar = tqdm(self.tr_loader, desc=f"Epoch {e}/{epochs} (eps={cur_eps:.3f}, steps={cur_steps})")
            for xb, yb in pbar:
                xb, yb = xb.to(self.device), yb.to(self.device)

                if adv_train:
                    xb, yb = adversarial_training_step(
                        self.model, xb, yb, attack=attack,
                        eps=cur_eps, step_size=pgd_step_size,
                        steps=cur_steps, lo=lo, hi=hi, frac=adv_frac
                    )

                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=amp):
                    logits = self.model(xb)
                    # label smoothing
                    if label_smooth > 0.0:
                        yb = yb * (1.0 - label_smooth) + 0.5 * label_smooth
                    loss = F.binary_cross_entropy_with_logits(
                        logits, yb, pos_weight=pos_weight
                    )

                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            acc, auc = self.evaluate()
            print(f"Val acc={acc:.4f} auc={auc:.4f}")
            if acc > best and ckpt:
                os.makedirs(os.path.dirname(ckpt), exist_ok=True)
                torch.save({'model_state': self.model.state_dict(),
                            'val_acc': acc, 'val_auc': auc}, ckpt)
                best = acc
                print(f"Saved -> {ckpt}")


    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set (clean)."""
        self.model.eval()
        ys, ps, probs = [], [], []

        for xb, yb in self.va_loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            logits = self.model(xb)
            prob = torch.sigmoid(logits)
            pred = (prob > 0.5).float()

            ys.append(yb.cpu().numpy())
            ps.append(pred.cpu().numpy())
            probs.append(prob.detach().cpu().numpy())

        y = np.concatenate(ys)
        yhat = np.concatenate(ps)
        pr = np.concatenate(probs)

        from sklearn.metrics import accuracy_score, roc_auc_score
        acc = accuracy_score(y, yhat)
        try:
            auc = roc_auc_score(y, pr)
        except Exception:
            auc = float("nan")

        return acc, auc

    # NOTE: no @torch.no_grad() here — we need grads for crafting adversarial examples
    def eval_under_attack(self, attack='pgd', eps=0.1, steps=5, step_size=0.02, lo=None, hi=None):
        """Evaluate under adversarial attack."""
        from .attacks.fgsm_pgd import fgsm, pgd

        self.model.eval()
        ys, ps, probs = [], [], []

        for xb, yb in self.va_loader:
            xb, yb = xb.to(self.device), yb.to(self.device)

            # Craft adversarial batch (attacks manage their own grad context)
            if attack == "fgsm":
                xadv = fgsm(self.model, xb, yb, eps, lo, hi)
            elif attack == "pgd":
                xadv = pgd(self.model, xb, yb, eps, step_size, steps, lo, hi)
            else:
                raise ValueError("attack must be 'fgsm' or 'pgd'")

            # Evaluate on adversarial inputs without computing grads
            with torch.no_grad():
                logits = self.model(xadv)
                prob = torch.sigmoid(logits)
                pred = (prob > 0.5).float()

            ys.append(yb.cpu().numpy())
            ps.append(pred.cpu().numpy())
            probs.append(prob.detach().cpu().numpy())

        y = np.concatenate(ys)
        yhat = np.concatenate(ps)
        pr = np.concatenate(probs)

        from sklearn.metrics import accuracy_score, roc_auc_score
        acc = accuracy_score(y, yhat)
        try:
            auc = roc_auc_score(y, pr)
        except Exception:
            auc = float("nan")

        return acc, auc
