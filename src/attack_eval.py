import numpy as np
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, HopSkipJump
from art.estimators.classification import SklearnClassifier, TensorFlowV2Classifier
from art.utils import to_categorical
import joblib

def attack_with_surrogate_and_transfer(X_test, y_test, surrogate_model, target_model, eps=0.1):
    # Use ART's FGSM on surrogate (e.g., a TF/PyTorch model)
    # surrogate_model should be an ART estimator (e.g., TensorFlowV2Classifier)
    attack = FastGradientMethod(estimator=surrogate_model, eps=eps)
    x_adv = attack.generate(x=X_test)
    # Evaluate transferability against target (LightGBM)
    y_pred_clean = target_model.predict(X_test)
    y_pred_adv = target_model.predict(x_adv)
    transfer_success = np.mean(y_pred_clean != y_pred_adv)
    return x_adv, transfer_success

def hopskipjump_on_model(target_model, X_test, y_test):
    # Wrap LightGBM model with ART's SklearnClassifier or a custom wrapper
    art_estimator = SklearnClassifier(model=target_model)
    hs = HopSkipJump(classifier=art_estimator, max_iter=50)
    x_adv = hs.generate(x=X_test)
    y_pred_clean = target_model.predict(X_test)
    y_pred_adv = target_model.predict(x_adv)
    success = np.mean(y_pred_clean != y_pred_adv)
    return x_adv, success