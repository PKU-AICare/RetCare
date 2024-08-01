from sklearn.linear_model import LogisticRegression
import shap


class LR():
    def __init__(self, **params):
        """params is a dict
        seed: int, random seed
        n_estimators: int, number of trees
        max_depth: int, depth of trees
        """
        task = params['task']
        self.task = task
        seed = params['seed']
        self.model = LogisticRegression(random_state=seed)

    def fit(self, x, y):
        if self.task == "outcome":
            self.model.fit(x, y)
            self.explainer = shap.Explainer(self.model.predict_proba, x)
        else:
            raise ValueError("Task must be either 'outcome' or 'los'.")
    
    def predict(self, x):
        if self.task == "outcome":
            res = self.model.predict_proba(x)
            if len(res.shape) == 1:
                return res[1]
            else:
                return res[:, 1]
        else:
            raise ValueError("Task must be either 'outcome' or 'los'.")

    def get_feature_importance(self, x, importance_type="built-in"):
        """Get feature importance of the model

        Args:
            importance_type (str, optional): Type of importance. Defaults to 'built-in'. 'built-in' or 'shap'

        Returns:
            _type_: list or np.array
        """
        if importance_type == "built-in":
            return self.model.feature_importances_
        elif importance_type == "shap":
            shap_values = self.explainer.shap_values(x)
            return shap_values