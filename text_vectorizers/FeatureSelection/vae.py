import torch
import torch.nn as nn

class VAE(nn.Module):
    """
    Класс для реализации Variational Autoencoder (VAE).
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 32, activation: callable = torch.sigmoid):
        """
        Инициализация класса.

        input_dim: int
                   Размерность входных данных.
        latent_dim: int
                    Размерность вектора признаков
        hidden_dim: int
                    Размерности скрытого слоя.
        activation: callable
                    Функция активации, по-умолчанию сигмоидальная
        """
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        self.activation = activation

    def encode(self, x: torch.Tensor):
        """
        Кодирование данных
        x: torch.Tensor
           Входные данные.
        """
        h1 = torch.relu(self.fc1(x))
        return self.fc2_mu(h1), self.fc2_logvar(h1)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """
        Репараметризация для расчета случайной величины.
        mu: torch.Tensor
            Среднее значение латентного пространства.
        logvar: torch.Tensor
                Логарифм дисперсии латентного пространства.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z:torch.Tensor):
        """
        Декодирование данных.
        z: torch.Tensor
           Вектор латентного пространства.
        """
        h3 = torch.relu(self.fc3(z))
        return self.activation(self.fc4(h3))

    def forward(self, x: torch.Tensor):
        """
        Прямой проход через VAE
        x: torch.Tensor
           Входные данные.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
        

def loss_function(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, loss_func: callable = nn.functional.binary_cross_entropy, reduction:str = 'sum'):
    """
    Функция потерь для VAE.
    recon_x: torch.Tensor
             Восстановленные данные
    x: torch.Tensor
       Исходные данные
    mu: torch.Tensor
        Средний вектор латентного пространства.
    logvar: torch.Tensor
            Логарифм дисперсии латентного пространства.
    loss_func: callable
               Функция потерь, по умолчанию BCE
    reduction: str
               Метод вычисления потерь.
    """

    # Функция потерь для VAE
    loss_rec = loss_func(recon_x, x, reduction=reduction)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return loss_rec + KLD