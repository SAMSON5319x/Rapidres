import os
import joblib
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

class LGBTrainer:
    def __init__(self):
        self.model = lgb.LGBMClassifier(objective='binary', n_estimators=500,
                                        learning_rate=0.05, num_leaves=64,
                                        random_state=42, n_jobs=-1)
    def fit(self, X_tr, y_tr, X_va, y_va, out_path=None):
        self.model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=50, verbose=50)
        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            joblib.dump(self.model, out_path)
        return self
    def evaluate(self, X, y):
        yhat = self.model.predict(X); proba = self.model.predict_proba(X)[:,1]
        acc = accuracy_score(y, yhat); auc = roc_auc_score(y, proba)
        print(classification_report(y, yhat)); print(confusion_matrix(y, yhat))
        return acc, auc
