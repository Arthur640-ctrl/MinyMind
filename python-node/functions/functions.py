import numpy as np
import matplotlib.pyplot as plt

class Activation:
    def __init__(self, activation="identity"):
        # on stocke le nom de l’activation choisie
        self.activation_name = activation.lower()

    def activation(self, x):
        """Applique la fonction d'activation choisie"""
        if self.activation_name == "identity":
            return self.identity(x)
        elif self.activation_name == "relu":
            return self.relu(x)
        elif self.activation_name == "sigmoid":
            return self.sigmoid(x)
        elif self.activation_name == "tanh":
            return self.tanh(x)
        elif self.activation_name == "leaky_relu":
            return self.leaky_relu(x)
        elif self.activation_name == "elu":
            return self.elu(x)
        elif self.activation_name == "softplus":
            return self.softplus(x)
        elif self.activation_name == "swish":
            return self.swish(x)
        elif self.activation_name == "mish":
            return self.mish(x)
        else:
            raise ValueError(f"Activation '{self.activation_name}' inconnue.")

    def derivative(self, x):
        """Applique la dérivée de la fonction choisie"""
        if self.activation_name == "identity":
            return self.d_identity(x)
        elif self.activation_name == "relu":
            return self.d_relu(x)
        elif self.activation_name == "sigmoid":
            return self.d_sigmoid(x)
        elif self.activation_name == "tanh":
            return self.d_tanh(x)
        elif self.activation_name == "leaky_relu":
            return self.d_leaky_relu(x)
        elif self.activation_name == "elu":
            return self.d_elu(x)
        elif self.activation_name == "softplus":
            return self.d_softplus(x)
        elif self.activation_name == "swish":
            return self.d_swish(x)
        elif self.activation_name == "mish":
            return self.d_mish(x)
        else:
            raise ValueError(f"Dérivée pour '{self.activation_name}' inconnue.")

    # Identity
    def identity(self, x):
        return x

    def d_identity(self, x):
        return np.ones_like(x)
    # Identity : la sortie est la même que l’entrée, dérivée = 1 partout.


    # ReLU
    def relu(self, x):
        return np.maximum(0, x)

    def d_relu(self, x):
        return np.where(x > 0, 1, 0)
    # ReLU : met à 0 les valeurs négatives, garde les positives.


    # Sigmoid
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def d_sigmoid(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    # Sigmoid : transforme les valeurs en [0,1], utile pour probabilités.


    # Tanh
    def tanh(self, x):
        return np.tanh(x)

    def d_tanh(self, x):
        return 1 - np.tanh(x) ** 2
    # Tanh : sortie entre -1 et 1, centrée.


    # Leaky ReLU
    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def d_leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)
    # Leaky ReLU : comme ReLU mais laisse passer un peu les valeurs négatives.


    # ELU
    def elu(self, x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    def d_elu(self, x, alpha=1.0):
        return np.where(x > 0, 1, alpha * np.exp(x))
    # ELU : améliore ReLU en gardant de la continuité pour x<0.


    # Softplus
    def softplus(self, x):
        return np.log1p(np.exp(x))

    def d_softplus(self, x):
        return self.sigmoid(x)
    # Softplus : version "douce" de ReLU.


    # Swish
    def swish(self, x):
        return x * self.sigmoid(x)

    def d_swish(self, x):
        s = self.sigmoid(x)
        return s + x * s * (1 - s)
    # Swish : activation lisse, souvent meilleure que ReLU.


    # Mish
    def mish(self, x):
        return x * np.tanh(np.log1p(np.exp(x)))

    def d_mish(self, x, eps=1e-5):
        # dérivée numérique approximative
        return (self.mish(x + eps) - self.mish(x - eps)) / (2 * eps)
    # Mish : comme Swish, mais encore plus lisse.

# x = np.linspace(-10, 10, 100)

# activations = [
#     "identity", "relu", "sigmoid", "tanh",
#     "leaky_relu", "elu", "softplus", "swish", "mish"
# ]

# plt.figure(figsize=(12, 8))

# for act in activations:
#     a = Activation(act)
#     y = a.activation(x)
#     plt.plot(x, y, label=act)

# plt.title("Fonctions d’activation")
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.legend()
# plt.grid(True)
# plt.show()

# x = np.linspace(-5, 5, 500)

# act = Activation("relu")
# y = act.activation(x)
# dy = act.derivative(x)

# plt.plot(x, y, label="ReLU")
# plt.plot(x, dy, label="dReLU")
# plt.legend()
# plt.show()
