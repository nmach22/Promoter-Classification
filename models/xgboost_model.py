import xgboost as xgb
import numpy as np

class XGBoostPromoterModel:
    """
    Wrapper for XGBoost classifier to handle promoter classification.
    """
    def __init__(self, **kwargs):
        """
        Initialize XGBClassifier with default or provided parameters.
        Default parameters are tuned for this task based on the notebook.
        """
        default_params = {
            'objective': 'binary:logistic',
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'logloss',
            'early_stopping_rounds': 20,
            'random_state': 42,
            'n_jobs': -1
        }
        # Update defaults with any provided kwargs
        default_params.update(kwargs)
        
        self.model = xgb.XGBClassifier(**default_params)

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=50):
        """
        Train the model.
        """
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=verbose
        )

    def predict(self, X):
        """
        Predict class labels.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities.
        """
        return self.model.predict_proba(X)

    def save_model(self, path):
        """
        Save the model to a file.
        """
        self.model.save_model(path)

    def load_model(self, path):
        """
        Load the model from a file.
        """
        self.model.load_model(path)
