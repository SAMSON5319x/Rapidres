# src/main.py
from data_loader import prepare_data
from train_baseline import train_lightgbm, evaluate
from attack_eval import hopskipjump_on_model, attack_with_surrogate_and_transfer
from defenses import adversarial_training_retrain
import joblib
import numpy as np

# 1: Prepare
X_train, X_test, y_train, y_test, scaler = prepare_data('../data/ember_train.csv', test_csv='../data/ember_test.csv')

# 2: Train baseline
model = train_lightgbm(X_train, y_train, X_test, y_test, output_model_path='models/lgb_baseline.pkl')
baseline_metrics = evaluate(model, X_test, y_test)

# 3: Attack (decision-based)
x_adv, success = hopskipjump_on_model(model, X_test[:2000], y_test[:2000])
print("Attack success (decision-based):", success)

# 4: Adversarial training (retrain on a subset)
y_adv = model.predict(x_adv)  # labels for adv samples (or original y)
model_defended = adversarial_training_retrain(model, X_train[:50000], y_train[:50000], x_adv, y_test[:x_adv.shape[0]], retrain_model_path='models/lgb_defended.pkl')

# 5: Evaluate defended
defended_metrics = evaluate(model_defended, X_test, y_test)
