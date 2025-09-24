"""
Reseau de neurones from scratch (entierement personalisable)

Requirements:
    pip install numpy

Instructions:
    - Les noms de variables sont en francais et tres explicites.
    - Les commentaires sont en francais tres simples (niveau enfant).
    - Le reseau supporte activations par couche ou une activation "globale".
    - On peut choisir l'optimiseur: "sgd", "momentum", "rmsprop", "adam", ou None.
    - Exemple d'entraînement a la fin du fichier.
"""

import numpy as np

# -------------------------
# Fonctions d'activation
# -------------------------
class FonctionsActivation:
    """Collection de fonctions d'activation et de leurs derivees."""

    @staticmethod
    def lineaire(x):
        # f(x) = x
        return x

    @staticmethod
    def derivee_lineaire(x):
        return np.ones_like(x)

    @staticmethod
    def relu(x):
        # f(x) = max(0,x)
        return np.maximum(0, x)

    @staticmethod
    def derivee_relu(x):
        # derivee: 1 si x>0, sinon 0
        grad = np.zeros_like(x)
        grad[x > 0] = 1.0
        return grad

    @staticmethod
    def sigmoid(x):
        # f(x) = 1 / (1 + e^-x)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def derivee_sigmoid(x):
        s = FonctionsActivation.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def derivee_tanh(x):
        t = np.tanh(x)
        return 1 - t * t

    @staticmethod
    def gelu(x):
        # Approximation GELU: x * 0.5 * (1 + tanh( sqrt(2/pi) * (x + 0.044715 x^3) ))
        facteur = np.sqrt(2.0 / np.pi)
        return 0.5 * x * (1.0 + np.tanh(facteur * (x + 0.044715 * (x ** 3))))

    @staticmethod
    def derivee_gelu(x):
        # derivee approximative: on differentiate l'approximation (un peu longue mais ok)
        facteur = np.sqrt(2.0 / np.pi)
        tanh_arg = facteur * (x + 0.044715 * x ** 3)
        tanh_val = np.tanh(tanh_arg)
        sech2 = 1.0 - tanh_val ** 2
        term1 = 0.5 * (1.0 + tanh_val)
        term2 = 0.5 * x * sech2 * facteur * (1.0 + 3.0 * 0.044715 * x ** 2)
        return term1 + term2

    @staticmethod
    def swish(x):
        # f(x) = x * sigmoid(x)
        return x * FonctionsActivation.sigmoid(x)

    @staticmethod
    def derivee_swish(x):
        s = FonctionsActivation.sigmoid(x)
        return s + x * s * (1 - s)

    @staticmethod
    def softmax(x):
        # softmax stable sur l'axe des colonnes (derniere dimension)
        x_stable = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_stable)
        somme = np.sum(exp_x, axis=1, keepdims=True)
        return exp_x / somme

    # mapping de noms vers fonctions pour utilisation facile
    nom_vers_fonction = {
        "lineaire": (lineaire, derivee_lineaire),
        "relu": (relu, derivee_relu),
        "sigmoid": (sigmoid, derivee_sigmoid),
        "tanh": (tanh, derivee_tanh),
        "gelu": (gelu, derivee_gelu),
        "swish": (swish, derivee_swish),
        # softmax est traitée special (utilisée souvent en sortie + cross-entropy)
        "softmax": (softmax, None),
    }

# -------------------------
# Couche Dense (entièrement connectée)
# -------------------------
class CoucheDense:
    """Une couche dense: poids, biais et activation."""

    def __init__(self, nombre_entrees, nombre_neurones, nom_activation="relu"):
        # initialisation Xavier (Glorot) pour poids
        limite = np.sqrt(6.0 / (nombre_entrees + nombre_neurones))
        self.poids = np.random.uniform(-limite, limite, size=(nombre_entrees, nombre_neurones))
        self.biais = np.zeros((1, nombre_neurones))

        # nom de l'activation (string) pour cette couche
        self.nom_activation = nom_activation

        # memoire pour backward
        self.dernier_entree = None
        self.dernier_z = None  # valeur avant activation (z = xW + b)
        self.dernier_activation = None

        # pour optimiseur: stocker les gradients calcules
        self.gradient_poids = np.zeros_like(self.poids)
        self.gradient_biais = np.zeros_like(self.biais)

# -------------------------
# Optimiseurs simples
# -------------------------
class Optimiseur:
    """
    Optimiseur simple qui sait mettre a jour les poids et biais de toutes les couches.
    Supporte: sgd, momentum, rmsprop, adam, ou None (dans ce cas on fait une simple descente)
    """

    def __init__(self, type_optimiseur="sgd", taux_apprentissage=0.01, momentum=0.9, rho=0.9, epsilon=1e-8, beta1=0.9, beta2=0.999):
        self.type_optimiseur = type_optimiseur
        self.taux_apprentissage = taux_apprentissage
        self.momentum = momentum
        self.rho = rho
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

        # états pour chaque paramètre: on va stocker dans des listes paralleles
        self.etats_vitesse_poids = []  # pour momentum / adam (m)
        self.etats_vitesse_biais = []
        self.etats_carré_poids = []    # pour rmsprop / adam (v)
        self.etats_carré_biais = []
        self.pas_adam = 0

    def initialiser_etats(self, liste_couches):
        # appelle une seule fois pour créer tous les etats a zeros
        self.etats_vitesse_poids = []
        self.etats_vitesse_biais = []
        self.etats_carré_poids = []
        self.etats_carré_biais = []
        for couche in liste_couches:
            self.etats_vitesse_poids.append(np.zeros_like(couche.poids))
            self.etats_vitesse_biais.append(np.zeros_like(couche.biais))
            self.etats_carré_poids.append(np.zeros_like(couche.poids))
            self.etats_carré_biais.append(np.zeros_like(couche.biais))
        self.pas_adam = 0

    def mise_a_jour(self, liste_couches):
        # si pas d'etats, initialiser
        if not self.etats_vitesse_poids:
            self.initialiser_etats(liste_couches)

        self.pas_adam += 1

        for indice, couche in enumerate(liste_couches):
            grad_p = couche.gradient_poids
            grad_b = couche.gradient_biais

            if self.type_optimiseur is None or self.type_optimiseur == "sgd":
                # simple descente de gradient
                couche.poids = couche.poids - self.taux_apprentissage * grad_p
                couche.biais = couche.biais - self.taux_apprentissage * grad_b

            elif self.type_optimiseur == "momentum":
                v_p = self.etats_vitesse_poids[indice]
                v_b = self.etats_vitesse_biais[indice]
                v_p_new = self.momentum * v_p - self.taux_apprentissage * grad_p
                v_b_new = self.momentum * v_b - self.taux_apprentissage * grad_b
                couche.poids = couche.poids + v_p_new
                couche.biais = couche.biais + v_b_new
                self.etats_vitesse_poids[indice] = v_p_new
                self.etats_vitesse_biais[indice] = v_b_new

            elif self.type_optimiseur == "rmsprop":
                s_p = self.etats_carré_poids[indice]
                s_b = self.etats_carré_biais[indice]
                s_p_new = self.rho * s_p + (1 - self.rho) * (grad_p ** 2)
                s_b_new = self.rho * s_b + (1 - self.rho) * (grad_b ** 2)
                couche.poids = couche.poids - (self.taux_apprentissage * grad_p) / (np.sqrt(s_p_new) + self.epsilon)
                couche.biais = couche.biais - (self.taux_apprentissage * grad_b) / (np.sqrt(s_b_new) + self.epsilon)
                self.etats_carré_poids[indice] = s_p_new
                self.etats_carré_biais[indice] = s_b_new

            elif self.type_optimiseur == "adam":
                m_p = self.etats_vitesse_poids[indice]
                m_b = self.etats_vitesse_biais[indice]
                v_p = self.etats_carré_poids[indice]
                v_b = self.etats_carré_biais[indice]

                m_p_new = self.beta1 * m_p + (1 - self.beta1) * grad_p
                m_b_new = self.beta1 * m_b + (1 - self.beta1) * grad_b
                v_p_new = self.beta2 * v_p + (1 - self.beta2) * (grad_p ** 2)
                v_b_new = self.beta2 * v_b + (1 - self.beta2) * (grad_b ** 2)

                # biais-correction
                m_p_corrige = m_p_new / (1 - self.beta1 ** self.pas_adam)
                m_b_corrige = m_b_new / (1 - self.beta1 ** self.pas_adam)
                v_p_corrige = v_p_new / (1 - self.beta2 ** self.pas_adam)
                v_b_corrige = v_b_new / (1 - self.beta2 ** self.pas_adam)

                couche.poids = couche.poids - (self.taux_apprentissage * m_p_corrige) / (np.sqrt(v_p_corrige) + self.epsilon)
                couche.biais = couche.biais - (self.taux_apprentissage * m_b_corrige) / (np.sqrt(v_b_corrige) + self.epsilon)

                self.etats_vitesse_poids[indice] = m_p_new
                self.etats_vitesse_biais[indice] = m_b_new
                self.etats_carré_poids[indice] = v_p_new
                self.etats_carré_biais[indice] = v_b_new

            else:
                raise ValueError("Optimiseur inconnu: " + str(self.type_optimiseur))

# -------------------------
# Reseau de neurones
# -------------------------
class ReseauNeuronal:
    """Reseau faisant forward/backward et entrainement."""

    def __init__(self, tailles_couches, activation_par_couche=None, activation_globale="relu", optimiseur_type="sgd", taux_apprentissage=0.01):
        """
        tailles_couches: liste d'entiers, par ex. [2, 8, 8, 1]
            - le premier est la dimension d'entree
            - le dernier est la dimension de sortie
        activation_par_couche: None ou liste de noms (longueur = nombre de couches -1)
            - si None, on utilise activation_globale pour toutes les couches sauf la sortie
            - on accepte "softmax" pour la couche de sortie si classification
        activation_globale: nom de l'activation a utiliser par defaut
        optimiseur_type: "sgd", "momentum", "rmsprop", "adam", ou None
        """
        self.taille_par_couche = tailles_couches
        self.liste_couches = []

        # construire les couches
        for i in range(len(tailles_couches) - 1):
            entrees = tailles_couches[i]
            sorties = tailles_couches[i + 1]
            if activation_par_couche is not None:
                nom_activation = activation_par_couche[i]
            else:
                # si c'est la derniere couche et activation_globale=="softmax", laisser "softmax"
                if (i == len(tailles_couches) - 2) and (activation_globale == "softmax"):
                    nom_activation = "softmax"
                else:
                    nom_activation = activation_globale
            couche = CoucheDense(entrees, sorties, nom_activation=nom_activation)
            self.liste_couches.append(couche)

        # optimiseur
        self.optimiseur = Optimiseur(taux_apprentissage=taux_apprentissage)

    # ---------- forward ----------
    def propagation_avant(self, entree_batche):
        """
        entree_batche: shape (batch_size, input_dim)
        Retourne la sortie du reseau (batch_size, sortie_dim)
        """
        activation = entree_batche
        for couche in self.liste_couches:
            # calcul z = xW + b
            z = np.dot(activation, couche.poids) + couche.biais
            couche.dernier_entree = activation
            couche.dernier_z = z

            # appliquer activation
            nom_act = couche.nom_activation
            if nom_act == "softmax":
                sortie = FonctionsActivation.softmax(z)
                # pour softmax on ne peut pas calculer la derivee ici simplement;
                # on gere specialement dans le backprop si on utilise cross-entropy
            else:
                func = FonctionsActivation.nom_vers_fonction.get(nom_act, None)
                if func is None:
                    raise ValueError("Activation inconnue: " + str(nom_act))
                sortie = func[0](z)

            couche.dernier_activation = sortie
            activation = sortie

        return activation

    # ---------- loss ----------
    @staticmethod
    def perte_mse(prediction, cible):
        # mean squared error (regression)
        diff = prediction - cible
        return np.mean(0.5 * (diff ** 2))

    @staticmethod
    def derivée_perte_mse(prediction, cible):
        # derivee par rapport a prediction
        return (prediction - cible) / prediction.shape[0]

    @staticmethod
    def perte_categorie_crossentropy(prediction_probabilites, indices_classes_vraies):
        """
        prediction_probabilites: shape (batch, classes) -> sortie softmax
        indices_classes_vraies: shape (batch,) entiers avec classe vraie
        """
        # petite protection pour log
        petites_valeurs = 1e-12
        probs = np.clip(prediction_probabilites, petites_valeurs, 1.0 - petites_valeurs)
        log_probs = -np.log(probs[np.arange(len(probs)), indices_classes_vraies])
        return np.mean(log_probs)

    # ---------- backward ----------
    def retropropagation(self, sortie_res, cible, type_loss="mse", indices_classes_vraies=None):
        """
        sortie_res: sortie du reseau (batch, sortie_dim)
        cible: si type_loss == "mse" -> meme shape que sortie_res
               si type_loss == "crossentropy" -> on utilise indices_classes_vraies
        indices_classes_vraies: tableau des labels (batch,) si crossentropy
        """
        # on calcule gradient de la couche de sortie en fonction du loss choisi
        nombre_exemples = sortie_res.shape[0]

        # initialiser gradient pour la sortie
        if type_loss == "mse":
            gradient_sortie = self.derivée_perte_mse(sortie_res, cible)
            # gradient_sortie shape = (batch, sortie_dim)
        elif type_loss == "crossentropy":
            # on suppose que la couche de sortie a utilisé softmax
            # gradient pour softmax + cross-entropy simplifie: (y_pred - y_onehot)/N
            if indices_classes_vraies is None:
                raise ValueError("Pour crossentropy il faut indices_classes_vraies.")
            y_pred = sortie_res  # sortie_res doit être la sortie softmax
            y_onehot = np.zeros_like(y_pred)
            y_onehot[np.arange(nombre_exemples), indices_classes_vraies] = 1.0
            gradient_sortie = (y_pred - y_onehot) / nombre_exemples
        else:
            raise ValueError("Loss inconnue: " + str(type_loss))

        # remonter couche par couche
        gradient_entree = gradient_sortie
        # parcourir couches en sens inverse
        for idx in reversed(range(len(self.liste_couches))):
            couche = self.liste_couches[idx]
            nom_act = couche.nom_activation

            # si activation est softmax et loss = crossentropy, le gradient a deja été calculé (cas special)
            if nom_act == "softmax" and type_loss == "crossentropy":
                grad_z = gradient_entree
            else:
                # calcul dphi/dz : derivee de l'activation evaluate en dernier_z
                if nom_act == "softmax":
                    # softmax sans crossentropy (rare) -> on calcule numeriquement la derivee generique
                    # pour simplification on utilise la forme jacobienne approximée lente:
                    soft = FonctionsActivation.softmax(couche.dernier_z)
                    grad_z = np.zeros_like(soft)
                    for i in range(soft.shape[0]):
                        s = soft[i].reshape(-1, 1)
                        jac = np.diagflat(s) - np.dot(s, s.T)
                        grad_z[i] = np.dot(jac, gradient_entree[i])
                else:
                    func = FonctionsActivation.nom_vers_fonction.get(nom_act, None)
                    if func is None or func[1] is None:
                        raise ValueError("Derivee introuvable pour l'activation: " + str(nom_act))
                    derivee_activation = func[1](couche.dernier_z)
                    grad_z = gradient_entree * derivee_activation

            # gradient des paramètres: dL/dW = X^T dot grad_z
            entree_prev = couche.dernier_entree  # (batch, input_dim)
            grad_poids = np.dot(entree_prev.T, grad_z)  # (input_dim, output_dim)
            grad_biais = np.sum(grad_z, axis=0, keepdims=True)  # (1, output_dim)

            # sauvegarder gradients dans la couche (pour optimiseur)
            couche.gradient_poids = grad_poids
            couche.gradient_biais = grad_biais

            # calculer gradient pour la couche precedente (pour propager encore plus haut)
            gradient_entree = np.dot(grad_z, couche.poids.T)

    # ---------- entrainement ----------
    def entrainer(self, donnees_entree, cibles, epochs=100, batch_size=32, type_loss="mse", indices_classes_vraies=None, afficher_tous_les=10):
        """
        Entraine le reseau.
        - donnees_entree: (N, input_dim)
        - cibles: si MSE -> (N, sortie_dim)
                 si CrossEntropy -> ignored (utiliser indices_classes_vraies)
        - indices_classes_vraies: pour crossentropy (N,)
        """
        n = donnees_entree.shape[0]
        # initialiser etats d'optimiseur
        self.optimiseur.initialiser_etats(self.liste_couches)

        for epoch in range(1, epochs + 1):
            # melanger les donnees
            indices = np.arange(n)
            np.random.shuffle(indices)
            donnees_melangees = donnees_entree[indices]
            if type_loss == "mse":
                cibles_melangees = cibles[indices]
            else:
                cibles_melangees = None
            if indices_classes_vraies is not None:
                labels_melanges = indices_classes_vraies[indices]
            else:
                labels_melanges = None

            # mini-batch
            for debut in range(0, n, batch_size):
                fin = min(debut + batch_size, n)
                batch_x = donnees_melangees[debut:fin]
                if type_loss == "mse":
                    batch_y = cibles_melangees[debut:fin]
                else:
                    batch_y = None
                if labels_melanges is not None:
                    batch_labels = labels_melanges[debut:fin]
                else:
                    batch_labels = None

                sortie = self.propagation_avant(batch_x)

                # retropropagation
                if type_loss == "mse":
                    self.retropropagation(sortie, batch_y, type_loss="mse")
                else:
                    # ici on suppose que sortie est softmax et on a des labels
                    self.retropropagation(sortie, None, type_loss="crossentropy", indices_classes_vraies=batch_labels)

                # mise a jour des poids par optimiseur
                self.optimiseur.mise_a_jour(self.liste_couches)

            # calcul et affichage de la perte en fin d'epoch
            sortie_epoch = self.propagation_avant(donnees_entree)
            if type_loss == "mse":
                perte = self.perte_mse(sortie_epoch, cibles)
            else:
                perte = self.perte_categorie_crossentropy(sortie_epoch, indices_classes_vraies)
            if epoch % afficher_tous_les == 0 or epoch == 1 or epoch == epochs:
                print(f"Epoch {epoch}/{epochs}  -  Perte: {perte:.6f}")

    # ---------- prediction ----------
    def predire(self, donnees_entree):
        sortie = self.propagation_avant(donnees_entree)
        return sortie

# -------------------------
# Exemple d'utilisation
# -------------------------
# Exemple 2: Classification XOR (2D -> 2 classes)
print("\n=== Exemple classification XOR ===")
# donnees XOR
entrees_xor = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)
labels_xor = np.array([0, 1, 1, 0])  # classes 0 ou 1

# reseau: 2 entrées -> 8 -> 8 -> 2 sorties (softmax)
reseau_xor = ReseauNeuronal(
    tailles_couches=[2, 8, 8, 2],
    activation_par_couche=["tanh", "tanh", "softmax", "linear"], 
    optimiseur_type="adam",
    taux_apprentissage=0.01
)

reseau_xor.entrainer(
    entrees_xor,
    None,
    epochs=500,
    batch_size=4,
    type_loss="crossentropy",
    indices_classes_vraies=labels_xor,
    afficher_tous_les=100
)

# predictions
sortie_xor = reseau_xor.predire(entrees_xor)
classes_predites = np.argmax(sortie_xor, axis=1)
for i in range(4):
    print(f"input={entrees_xor[i]}  label={labels_xor[i]}  pred={classes_predites[i]}  probs={sortie_xor[i]}")

