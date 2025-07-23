from sklearn.calibration import LabelEncoder
from xgboost import XGBClassifier


class XGBClassifierWithLabelEncoder(XGBClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._label_encoder = LabelEncoder()
        self._classes = None

    def fit(self, X, y, **kwargs):
        y_encoded = self._label_encoder.fit_transform(y)
        self._classes = self._label_encoder.classes_
        return super().fit(X, y_encoded, **kwargs)

    def predict(self, X, **kwargs):
        y_pred_encoded = super().predict(X, **kwargs)
        return self._label_encoder.inverse_transform(y_pred_encoded)

    def predict_proba(self, X, **kwargs):
        return super().predict_proba(X, **kwargs)

    def score(self, X, y, **kwargs):
        y_encoded = self._label_encoder.transform(y)
        return super().score(X, y_encoded, **kwargs)