import random
from activations import *

class NeuralNetwork:
    def __init__(self, model_size, neurone_initialization=True, activation="linear"):
        # Taille du modèle : liste qui spécifie le nombre de neurones par couche
        # Exemple : [2, 3, 1] = 2 entrées, 3 neurones cachés, 1 sortie
        self.model_size = model_size
        
        # Si True, initialise les poids aléatoirement entre -0.1 et 0.1
        # Si False, initialise les poids à 0
        self.neurone_initialization = neurone_initialization
        
        # Fonction d'activation utilisée pour toutes les couches
        self.activation = activation
        
        # Listes pour stocker les paramètres du réseau
        self.poids = []        # Poids des connexions entre les neurones
        self.biais = []        # Biais de chaque neurone
        self.deltas = []       # Deltas calculés pendant la backpropagation
        
        # Initialisation du réseau avec les poids et biais
        self.init_network()
    
    def f(self, x):
        """Applique la fonction d'activation à x"""
        if self.activation == "linear":
            return Linear.f(x)
        elif self.activation == "sigmoid":
            return Sigmoid.f(x)
        elif self.activation == "tanh":
            return Tanh.f(x)
        elif self.activation == "relu":
            return ReLU.f(x)
        elif self.activation == "softmax":
            return Softmax.f(x)
    
    def df(self, x):
        """Calcule la dérivée de la fonction d'activation en x"""
        if self.activation == "linear":
            return Linear.d(x)
        elif self.activation == "sigmoid":
            return Sigmoid.d(x)
        elif self.activation == "tanh":
            return Tanh.d(x)
        elif self.activation == "relu":
            return ReLU.d(x)
        elif self.activation == "softmax":
            return Softmax.d(x)

    def init_network(self):
        """Initialise les poids et biais du réseau neuronal"""
        
        # Initialisation des poids pour chaque couche (sauf la couche d'entrée)
        for couche_index in range(1, len(self.model_size)):
            # Nombre de neurones dans la couche actuelle
            nb_neurones_couche_actuelle = self.model_size[couche_index]
            # Nombre de neurones dans la couche précédente (pour les connexions)
            nb_neurones_couche_precedente = self.model_size[couche_index - 1]

            # Liste pour stocker les poids de tous les neurones de cette couche
            poids_couche_actuelle = []

            # Pour chaque neurone de la couche actuelle
            for neurone_index in range(nb_neurones_couche_actuelle):
                # Liste pour stocker les poids venant de tous les neurones de la couche précédente
                poids_neurone_actuel = []

                # Pour chaque connexion venant de la couche précédente
                for connexion_index in range(nb_neurones_couche_precedente):
                    if self.neurone_initialization:
                        # Initialisation aléatoire du poids
                        poids_neurone_actuel.append(random.uniform(-0.1, 0.1))
                    else:
                        # Initialisation à 0
                        poids_neurone_actuel.append(0)

                # Ajouter les poids de ce neurone à la liste de la couche
                poids_couche_actuelle.append(poids_neurone_actuel)

            # Ajouter tous les poids de cette couche à la liste globale des poids
            self.poids.append(poids_couche_actuelle)

        # Initialisation des biais pour chaque couche (sauf la couche d'entrée)
        for couche_index in range(1, len(self.model_size)):
            # Nombre de neurones dans la couche actuelle
            nb_neurones_couche_actuelle = self.model_size[couche_index]

            # Liste pour stocker les biais de tous les neurones de cette couche
            biais_couche_actuelle = []

            # Pour chaque neurone de la couche, on initialise son biais à 0
            for neurone_index in range(nb_neurones_couche_actuelle):
                biais_couche_actuelle.append(0)

            # Ajouter la liste des biais de cette couche à la liste globale des biais
            self.biais.append(biais_couche_actuelle)

    def forward(self, inputs):
        """Passe avant : calcule les sorties du réseau pour des entrées données"""
        
        # Les activations commencent par les entrées
        activations_courantes = inputs
        # On stocke toutes les activations de chaque couche pour la backpropagation
        activations_par_couche = [inputs]
        
        # Les valeurs pondérées avant l'application de la fonction d'activation
        valeurs_ponderees_par_couche = []
        
        # Calcul des activations couche par couche
        for index_couche in range(1, len(self.model_size)):
            # Liste pour les activations de la couche actuelle
            activations_couche_actuelle = []
            # Liste pour les valeurs pondérées de la couche actuelle
            valeurs_ponderees_couche_actuelle = []

            # Pour chaque neurone dans la couche actuelle
            for index_neurone in range(self.model_size[index_couche]):
                # Calcul de la somme pondérée (input * poids + biais)
                somme_ponderee = 0

                # Somme sur toutes les connexions de la couche précédente
                for index_entree in range(self.model_size[index_couche - 1]):
                    # Valeur d'entrée depuis la couche précédente
                    valeur_entree = activations_courantes[index_entree]
                    # Poids de la connexion entre le neurone d'entrée et le neurone actuel
                    poids_connexion = self.poids[index_couche - 1][index_neurone][index_entree]
                    # Contribution au calcul de la somme pondérée
                    somme_ponderee += valeur_entree * poids_connexion
                
                # Ajout du biais du neurone actuel
                somme_ponderee += self.biais[index_couche - 1][index_neurone]

                # Stockage de la valeur pondérée avant activation
                valeurs_ponderees_couche_actuelle.append(somme_ponderee)

                # Application de la fonction d'activation
                activation_neurone = self.f(somme_ponderee)
                activations_couche_actuelle.append(activation_neurone)

            # Stockage des valeurs pondérées de cette couche
            valeurs_ponderees_par_couche.append(valeurs_ponderees_couche_actuelle)

            # Les activations de cette couche deviennent les entrées de la suivante
            activations_courantes = activations_couche_actuelle
            activations_par_couche.append(activations_couche_actuelle)

        return activations_courantes, valeurs_ponderees_par_couche, activations_par_couche

    def loss(self, prediction, cible):
        """Calcule la dérivée de la fonction de perte (erreur quadratique)"""
        if isinstance(prediction, list):
            prediction = prediction[0]
        # Dérivée de (prediction - cible)² = 2*(prediction - cible)
        return 2 * (prediction - cible)

    def backward_propagation(self, sorties, valeurs_ponderees, cibles):
        """
        Passe arrière : calcule les deltas pour chaque neurone
        en propageant l'erreur depuis la sortie vers l'entrée
        """
        
        # Étape 1 : Calcul des deltas pour la couche de sortie
        deltas_sortie = []
        
        for index_sortie in range(len(sorties)):
            # Calcul de la dérivée de la perte pour cette sortie
            derivee_perte = self.loss(sorties[index_sortie], cibles[index_sortie])
            # Valeur pondérée correspondante avant activation
            valeur_ponderee_sortie = valeurs_ponderees[-1][index_sortie]
            # Delta = dérivée de la perte × dérivée de la fonction d'activation
            delta_sortie = derivee_perte * self.df(valeur_ponderee_sortie)
            deltas_sortie.append(delta_sortie)

        # Initialisation de la liste des deltas avec ceux de la couche de sortie
        deltas_par_couche = [deltas_sortie]

        # Étape 2 : Propagation de l'erreur vers les couches cachées
        # On parcourt les couches de l'avant-dernière à la première couche cachée
        for index_couche in range(len(self.model_size) - 2, 0, -1):
            deltas_couche_actuelle = []
            
            # Pour chaque neurone de la couche actuelle
            for index_neurone in range(self.model_size[index_couche]):
                # Récupération des poids de la couche suivante
                poids_couche_suivante = self.poids[index_couche]
                # Récupération des deltas de la couche suivante (déjà calculés)
                deltas_couche_suivante = deltas_par_couche[0]

                # Calcul de l'erreur propagée depuis la couche suivante
                erreur_propagee = 0
                for index_delta in range(len(deltas_couche_suivante)):
                    # Contribution de chaque neurone de la couche suivante
                    # Le poids connecte le neurone actuel au neurone de la couche suivante
                    erreur_propagee += deltas_couche_suivante[index_delta] * poids_couche_suivante[index_delta][index_neurone]
                
                # Récupération de la valeur pondérée du neurone actuel
                valeur_ponderee_neurone = valeurs_ponderees[index_couche - 1][index_neurone]
                # Calcul du delta : erreur propagée × dérivée de l'activation
                delta_neurone = erreur_propagee * self.df(valeur_ponderee_neurone)
                deltas_couche_actuelle.append(delta_neurone)

            # Insertion des deltas de la couche actuelle au début de la liste
            deltas_par_couche.insert(0, deltas_couche_actuelle)

        # Stockage des deltas pour la mise à jour des poids
        self.deltas = deltas_par_couche
        return deltas_par_couche

    def update_weights_biases(self, activations, taux_apprentissage=0.01):
        """Met à jour les poids et biais en utilisant les deltas calculés"""
        
        # Parcours de toutes les couches (sauf l'entrée)
        for index_couche in range(len(self.model_size) - 1):
            nombre_neurones_couche = self.model_size[index_couche + 1]
            nombre_entrees_couche = self.model_size[index_couche]

            # Pour chaque neurone de la couche actuelle
            for index_neurone in range(nombre_neurones_couche):
                # Mise à jour des poids des connexions entrantes
                for index_entree in range(nombre_entrees_couche):
                    # Valeur d'activation de la couche précédente
                    valeur_activation = activations[index_couche][index_entree]
                    # Delta correspondant à ce neurone
                    delta_neurone = self.deltas[index_couche][index_neurone]
                    # Mise à jour du poids : poids -= taux_apprentissage × delta × activation
                    self.poids[index_couche][index_neurone][index_entree] -= taux_apprentissage * delta_neurone * valeur_activation
                
                # Mise à jour du biais du neurone
                delta_neurone = self.deltas[index_couche][index_neurone]
                self.biais[index_couche][index_neurone] -= taux_apprentissage * delta_neurone

    def train(self, entrees, cibles, taux_apprentissage=0.01, iterations=1000, debug_progression_rate = 100):
        """Entraîne le réseau neuronal sur un ensemble de données"""
        
        # Vérification que les données d'entrée et cibles ont la même longueur
        if len(entrees) != len(cibles):
            raise ValueError("Le nombre d'entrées doit correspondre au nombre de cibles")
        
        # Boucle d'entraînement sur le nombre depoch donné en arg
        for iteration in range(iterations):
            erreur_totale = 0
            
            # Pour chaque exemple d'entraînement
            for entree, cible in zip(entrees, cibles):

                # 1. Forward : calcul des sorties
                sorties, valeurs_ponderees, activations_par_couche = self.forward(entree)
                
                # 2. Backward : calcul des deltas
                # On convertit la cible en liste, pour avoir les meme type de donné
                if not isinstance(cible, list):
                    cible = [cible]

                deltas = self.backward_propagation(sorties, valeurs_ponderees, cible)
                
                # 3. Mise à jour des poids et biais
                self.update_weights_biases(activations_par_couche, taux_apprentissage)
                                
                # Si l'entrée est une liste
                if isinstance(cible, list):
                    # Pour chaque neurone de sortie, on fait (prédiction - cible)²
                    erreur = 0
                    for i in range(len(sorties)):
                        difference = sorties[i] - cible[i]  # Différence entre prédiction et cible
                        erreur_au_carre = difference ** 2   # Élévation au carré
                        erreur += erreur_au_carre          # Addition des erreurs
                
                # Sinon si c'est juste un float/int 
                else:
                    difference = sorties[0] - cible         # Différence entre prédiction et cible
                    erreur = difference ** 2               # Élévation au carré
                
                # On ajoute l'erreur de cet exemple à l'erreur totale de l'itération
                erreur_totale += erreur
            
            if iteration % debug_progression_rate == 0:
                erreur_moyenne = erreur_totale / len(entrees)
                print(f"Itération {iteration}, Erreur moyenne: {erreur_moyenne:.6f}")
