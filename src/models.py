"""
Siulomas modelis: hibridinis (fizika + ML) ir asmeninis vairuotojo profilis.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge

from features import FEATURES, GEO_NAIVE, GEO_PHYS


class GeometricModel(BaseEstimator, RegressorMixin):
    """Analitinis metodas be mokymo. cols - GEO_NAIVE arba GEO_PHYS."""
    def __init__(self, cols="naive"):
        self.cols = cols

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X[GEO_NAIVE if self.cols == "naive" else GEO_PHYS].to_numpy()


class HybridMirrorModel(BaseEstimator, RegressorMixin):
    """Fizikinis ivertis + ML pataisa (Ridge regresija mokosi likutines paklaidos).

    Modelis mokosi ne viso kampo, o tik to, ko fizikinis modelis nepaaiskina
    (residual learning). Tai stabiliau, kai duomenu nedaug.
    """
    FEATS = FEATURES + GEO_PHYS

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.reg_ = Ridge(alpha=self.alpha).fit(X[self.FEATS], np.asarray(y) - X[GEO_PHYS].to_numpy())
        return self

    def predict(self, X):
        return X[GEO_PHYS].to_numpy() + self.reg_.predict(X[self.FEATS])


class DriverProfile:
    """Asmeninis profilis: vidutine vairuotojo rankiniu pataisymu paklaida (kiekvienam kampui).

    Kai vairuotojas pirma karta pats nustato veidrodelius, sistema isimena skirtuma tarp
    jo pasirinktu kampu ir modelio prognozes. Veliau sis poslinkis pridedamas automatiskai.
    shrink - kiek pataisymu "uztenka", kad poslinkiu butu pasitikima visiskai (Bajeso glodinimas).
    """
    def __init__(self, shrink=0.5):
        self.shrink = shrink
        self.sum = np.zeros(4)
        self.n = 0

    def add_correction(self, chosen_angles, predicted_angles):
        self.sum += np.asarray(chosen_angles) - np.asarray(predicted_angles)
        self.n += 1

    @property
    def offset(self):
        return self.sum / (self.n + self.shrink) if self.n else np.zeros(4)

    def apply(self, predicted_angles):
        return np.asarray(predicted_angles) + self.offset
