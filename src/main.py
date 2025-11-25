import argparse
import torch
from data_loader import prepare
from train_torch import TorchTrainer
from train_lgbm import LGBTrainer
from attacks.transfer import SurrogateTransfer
from models.mlp import MLP
from eval import append_run

parser = argparse.ArgumentParser()
parser.add_argument('--train_csv', required=True)
parser.add_argument('--test_csv', default=None)
parser.add_argument('--batch', type=int, default=4096)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--wd', type=float, default=1e-5)
parser.add_argument('--hidden', type=str, default='512,256,128')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--attack', choices=['fgsm','pgd'], default='pgd')
parser.add_argument('--eps', type=float, default=0.1)
parser.add_argument('--pgd_steps', type=int, default=5)
parser.add_argument('--pgd_step_size', type=float, default=0.02)
parser.add_argument('--adv_train', action='store_true')
parser.add_argument('--adv_frac', type=float, default=0.5)
parser.add_argument('--metrics_out', type=str, default='metrics/results.json')
args = parser.parse_args()

(X_tr, y_tr, X_va, y_va, scaler, x_min_np, x_max_np) = prepare(
    args.train_csv, args.test_csv, scaler_out='models/scaler.joblib')

hidden = tuple(int(x) for x in args.hidden.split(','))
TT = TorchTrainer(in_dim=X_tr.shape[1], hidden=hidden, dropout=args.dropout)
TT.loaders(X_tr, y_tr, X_va, y_va, batch=args.batch)

lo = torch.from_numpy(x_min_np).to(TT.device)
hi = torch.from_numpy(x_max_np).to(TT.device)

TT.fit(epochs=args.epochs, lr=args.lr, wd=args.wd, amp=True,
       adv_train=args.adv_train, attack=args.attack, eps=args.eps,
       pgd_steps=args.pgd_steps, pgd_step_size=args.pgd_step_size,
       adv_frac=args.adv_frac, lo=lo, hi=hi,
       ckpt='models/torch_defended.pt' if args.adv_train else 'models/torch_baseline.pt')

clean_acc, clean_auc = TT.evaluate()
rob_acc, rob_auc = TT.eval_under_attack(attack=args.attack, eps=args.eps,
                                        steps=args.pgd_steps, step_size=args.pgd_step_size,
                                        lo=lo, hi=hi)
append_run(args.metrics_out, 'torch_' + ('defended' if args.adv_train else 'baseline'),
           {'clean_acc': clean_acc, 'clean_auc': clean_auc,
            'rob_acc': rob_acc, 'rob_auc': rob_auc,
            'attack': args.attack, 'eps': args.eps})

LGB = LGBTrainer().fit(X_tr, y_tr, X_va, y_va, out_path='models/lgb_baseline.pkl')
acc_lgb, auc_lgb = LGB.evaluate(X_va, y_va)
append_run(args.metrics_out, 'lgb_baseline', {'clean_acc': acc_lgb, 'clean_auc': auc_lgb})

surr = MLP(in_dim=X_tr.shape[1], hidden=hidden)
surr.load_state_dict(TT.model.state_dict())
ST = SurrogateTransfer(surr)
X_adv, clean_acc_lgb, adv_acc_lgb, tsr = ST.craft_and_transfer(LGB.model, X_va, y_va,
                                                               attack=args.attack, eps=args.eps,
                                                               steps=args.pgd_steps, step_size=args.pgd_step_size)
append_run(args.metrics_out, 'transfer_to_lgb',
           {'clean_acc': clean_acc_lgb, 'adv_acc': adv_acc_lgb, 'transfer_success': tsr,
            'attack': args.attack, 'eps': args.eps})
print("\nSaved metrics ->", args.metrics_out)
