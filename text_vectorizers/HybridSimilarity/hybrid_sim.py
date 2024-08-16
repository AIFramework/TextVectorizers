import torch
import torch.nn as nn
import torch.optim as optim

class HybridSimilarityMetric:
    """
    Класс для реализации гибридной метрики схожести.
    """
    def __init__(self, extractor, hybrid_model, optimizer = optim.AdamW(self.model.parameters(), lr=0.001), loss = nn.BCELoss()):
      """
      Инициализация класса для реализации гибридной метрики схожести

      extractor: TextFeatureExtractor
                 Экземпляр класса для извлечения признаков из текстов
      hybrid_model: nn.Module
                    Модель нейронной сети для гибридной метрики схожести
      optimizer: torch.optim
                 Оптимизатор для обучения модели по умолчанию - AdamW с lr = 0.001
      loss: nn.Module
            Функция потерь для обучения модели по умолчанию - бинарная кросс-энтропия
      """

        self.model = hybrid_model
        # Функция потерь
        self.criterion = loss
        # Оптимизатор
        self.optimizer = optimizer
        self.extractor = extractor

    def fit(self, X, y, epochs=100, show_step = 10):
        """
        Обучаем модель на данных.

        X: np.array, форма (n_samples, n_features)
           Матрица признаков, где каждая строка представляет собой метрики для пары текстов.
        y: np.array, форма (n_samples,)
           Целевые значения принадлежности (в диапазоне от 0 до 1).
        epochs: int
            Количество эпох для обучения.
        """
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            if (epoch+1) % show_step == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    def predict_proba(self, X):
        """
        Предсказываем вероятность того, что пара текстов схожа.

        X: np.array, форма (n_samples, n_features)
           Матрица признаков для новых пар текстов.

        return: np.array, форма (n_samples,)
                Вероятности схожести для каждой пары текстов.
        """
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X)
        return outputs.numpy().flatten()

    def predict_vectors(self, X, threshold=0.5):
        """
        Предсказываем классы (схожи/несхожи) на основе заданного порога.

        X: np.array, форма (n_samples, n_features)
           Матрица признаков для новых пар текстов.
        threshold: float
                   Порог для определения классов (по умолчанию 0.5).

        return: np.array, форма (n_samples,)
                Предсказанные метки (1 — схожие тексты, 0 — несхожие тексты).
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def predict(self, texts):
        """
        Предсказываем вероятность того, что пара текстов схожа на основе текстов.

        text1: str
               Первый текст.
        text2: str
               Второй текст.
        extractor: TextFeatureExtractor
                   Экземпляр класса для извлечения признаков из текстов.

        return: float
                Вероятность схожести для пары текстов.
        """
        features = self.extractor.extract_features(texts[0][0], texts[0][1])
        return self.predict_proba(features.reshape(1, -1))

    def predict_w_v(self, texts, v1, v2):
        """
        Предсказываем вероятность того, что пара текстов схожа на основе текстов.

        text1: str
               Первый текст.
        text2: str
               Второй текст.
        extractor: TextFeatureExtractor
                   Экземпляр класса для извлечения признаков из текстов.

        return: float
                Вероятность схожести для пары текстов.
        """
        features = self.extractor.extract_features(texts[0][0], texts[0][1], v1, v2)
        return self.predict_proba(features.reshape(1, -1))