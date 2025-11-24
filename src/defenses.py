import numpy as np
from sklearn.utils import shuffle
import joblib

def adversarial_training_retrain(model, X_clean_train, y_train, X_adv, y_adv, retrain_model_path=None):
    X_combined = np.vstack([X_clean_train, X_adv])
    y_combined = np.concatenate([y_train, y_adv])
    # shuffle
    X_combined, y_combined = shuffle(X_combined, y_combined, random_state=42)
    model.fit(X_combined, y_combined)
    if retrain_model_path:
        joblib.dump(model, retrain_model_path)
    return model