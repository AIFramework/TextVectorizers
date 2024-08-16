import torch

import numpy as np
import torch.optim as optim
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler

from .FeatureSelection.vae import *

class VAEFeatureSelector:
    """
    Класс для отбора признаков с использованием VAE.
    """
    def __init__(self, n_features, latent_dim):
        """
        Инициализация класса.

        n_features: int
                    Количество признаков после уменьшения размерности.
        latent_dim: int
                    Размер латентного пространства VAE.
        """
        self.n_features = n_features
        self.latent_dim = latent_dim  # Размер латентного пространства VAE (latent_dim>=n_features)
        self.vae = None
        self.poly = PolynomialFeatures(degree=2, include_bias=False)  # Для полиномиального расширения
        self.scaler = MinMaxScaler()
        self.selected_features_indices = None  # Индексы выбранных признаков

    def fit(self, X_base, X_poly=None, epochs=100, lr=1e-3):
        """
        Обучение модели на данных.
        X_base: np.array
                Матрица признаков для базовых данных.
        X_poly: np.array
                Матрица признаков для расширенных данных.
        """
        # Применение полиномиального расширения к нужным признакам
        if X_poly is not None:
            X_poly = self.poly.fit_transform(X_poly)

        # Конкатенация двух массивов признаков
        if X_base is not None and X_poly is not None:
            X = np.hstack((X_base, X_poly))
        elif X_base is not None:
            X = X_base
        elif X_poly is not None:
            X = X_poly
        else:
            raise ValueError("Both X_base and X_poly cannot be None")

        # Нормализация признаков
        X = self.scaler.fit_transform(X)

        # Преобразование данных в тензоры
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Создание и обучение модели VAE
        input_dim = X_tensor.shape[1]
        self.vae = VAE(input_dim, self.latent_dim)
        optimizer = optim.Adam(self.vae.parameters(), lr=lr)

        for epoch in range(epochs):
            self.vae.train()
            optimizer.zero_grad()
            recon_batch, mu, logvar = self.vae(X_tensor)
            loss = loss_function(recon_batch, X_tensor, mu, logvar)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

        with torch.no_grad():
            _, mu, _ = self.vae(X_tensor)
            importance = torch.abs(mu).mean(dim=0).numpy()
            self.selected_features_indices = np.argsort(importance)[-self.n_features:]

    def transform(self, X_base, X_poly=None):
        """
        Применяет метод отбора признаков к данным.
        X_base: np.array
                Матрица признаков для базовых данных.
        X_poly: np.array
                Матрица признаков для расширенных данных.
        """
        if X_poly is not None:
            X_poly = self.poly.transform(X_poly)

        # Конкатенация двух массивов признаков
        if X_base is not None and X_poly is not None:
            X = np.hstack((X_base, X_poly))
        elif X_base is not None:
            X = X_base
        elif X_poly is not None:
            X = X_poly
        else:
            raise ValueError("Both X_base and X_poly cannot be None")

        # Нормализация признаков
        X = self.scaler.transform(X)

        # Преобразование данных в тензоры
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Преобразование данных с помощью обученной VAE
        with torch.no_grad():
            self.vae.eval()
            recon_batch, mu, logvar = self.vae(X_tensor)
            selected_features = mu[:, self.selected_features_indices].numpy()

        return selected_features