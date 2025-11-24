import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import joblib
import numpy as np

def train_lightgbm(X_train, y_train, X_val, y_val, output_model_path='models/lgb_baseline.pkl'):
    lgb_clf = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        random_state=42,
        n_jobs=-1
    )
    lgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=50)
    joblib.dump(lgb_clf, output_model_path)
    return lgb_clf

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print("Accuracy:", acc)
    print("ROC AUC:", auc)
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return {'acc': acc, 'auc': auc}