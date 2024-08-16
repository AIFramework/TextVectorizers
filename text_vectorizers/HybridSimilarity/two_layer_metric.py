import torch.nn as nn

class HybridSimilarityMetricNN(nn.Module):
    """
    Класс для реализации нейронной сети для гибридной метрики схожести.
    """
    def __init__(self, input_dim):
        """
        Инициализация класса.

        input_dim: int
                   Размерность входных данных.
        """
        super(HybridSimilarityMetricNN, self).__init__()
        h = input_dim*2
        self.layer1 = nn.Linear(input_dim, h)
        self.relu = nn.GELU()
        self.layer2 = nn.Linear(h, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        """
        Прямой проход через сеть.
        x: torch.Tensor
           Входные данные.
        """
        x = self.relu(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x