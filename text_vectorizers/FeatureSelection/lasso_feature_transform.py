from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures

class FeatureSelectorLasso:
    """
    Класс для отбора признаков с использованием Lasso-регуляризации
    """
    def __init__(self, n_features, lasso_alpha = 1e-2):
        """
        Инициализация класса

        n_features: int
                    Количество признаков после отбора
        """
        self.n_features = n_features
        self.scaler = MinMaxScaler()
        self.model = Lasso(alpha=lasso_alpha)
        self.selected_features_indices = None
        self.poly = None

    def fit(self, X, y):
        """
        Обучение модели на данных, матрица объекты-признаки и целевые метки
        X: np.array
           Матрица объекты-признаки
        y: np.array
           Целевые значения
        """
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = self.poly.fit_transform(X)

        self.model.fit(X_poly, y)
        importance = np.abs(self.model.coef_)

        self.selected_features_indices = np.argsort(importance)[-self.n_features:]

        X_selected = X_poly[:, self.selected_features_indices]
        self.scaler.fit(X_selected)

    def transform(self, X):
        """
        Применяет алгоритм отбора признаков к данным
        X: np.array
           Матрица объекты-признаки
        """
        X_poly = self.poly.transform(X)
        X_selected = X_poly[:, self.selected_features_indices]
        X_scaled = self.scaler.transform(X_selected)

        return X_scaled