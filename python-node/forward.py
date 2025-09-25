import random

class NeuralNetwork:
    def __init__(self, model_size, neurone_initialization=True):
        self.model_size = model_size
        self.neurone_initialization = neurone_initialization
        
        # Listes pour stocker les poids et les biais
        self.poids = []
        self.biais = []
        self.deltas = []
        
        # Initialisation du réseau
        self.init_network()
    
    def f(self, x):
        """Fonction linéaire (plus simple pour commencer)"""
        return x

    def df(self, x):
        """Dérivée de la fonction linéaire (toujours 1)"""
        return 1

    def init_network(self):
        # Génération des poids pour chaque couche (sauf la couche d'entrée)
        for couche_index in range(1, len(self.model_size)):
            # Nombre de neurones dans la couche actuelle
            nb_neurones_couche = self.model_size[couche_index]
            # Nombre de neurones dans la couche précédente (pour les connexions)
            nb_inputs_couche = self.model_size[couche_index - 1]

            # Liste pour stocker les poids de tous les neurones de cette couche
            poids_couche = []

            # Pour chaque neurone de la couche actuelle
            for neurone_index in range(nb_neurones_couche):
                # Liste pour stocker les poids venant de tous les neurones de la couche précédente
                poids_neurone = []

                # Pour chaque connexion venant de la couche précédente
                for connexion_index in range(nb_inputs_couche):
                    if self.neurone_initialization:
                        poids_neurone.append(random.uniform(-0.1, 0.1))
                    else:
                        poids_neurone.append(0)  # Initialisation du poids à 0

                # Ajouter les poids de ce neurone à la liste de la couche
                poids_couche.append(poids_neurone)

            # Ajouter tous les poids de cette couche à la liste globale des poids
            self.poids.append(poids_couche)

        print("Poids par couche :", self.poids)

        # Pour chaque couche du réseau, sauf la couche d'entrée
        for couche_index in range(1, len(self.model_size)):
            # Nombre de neurones dans la couche actuelle
            nb_neurones_couche = self.model_size[couche_index]

            # Liste pour stocker les biais de tous les neurones de cette couche
            biais_couche = []

            # Pour chaque neurone de la couche, on initialise son biais à 0
            for neurone_index in range(nb_neurones_couche):
                biais_couche.append(0)

            # Ajouter la liste des biais de cette couche à la liste globale des biais
            self.biais.append(biais_couche)

    def forward(self, inputs):
        activations = inputs
        activations_per_couche = [inputs]

        # Les valeurs pondérées avant la fonction d'activation
        weighted_values = []
        
        # Calcul des activations couche par couches
        for index_couche in range(1, len(self.model_size)):

            # Liste qui va contenir les résultats (activations) de la couche actuelle
            activations_couche = []

            # Valeurs pondérées pour chaque neurone de la couche
            weighted_couche = [] 

            # Pour chaque neuronne dans la couche actuelle :
            for index_neuronne in range(self.model_size[index_couche]):
                
                # 1. On calcule la somme pndéerée (input * poids + biais)
                somme_ponderee = 0

                # On addition les resulatats de chaque neurone de la couche précédante (si c la couche input, on recup uniquement l'input)
                for index_input in range(self.model_size[index_couche - 1]):
                    
                    valeur_input = activations[index_input] # L'entrée
                    poids_connexion = self.poids[index_couche - 1][index_neuronne][index_input] # Recuperaton du poids
                    
                    somme_ponderee += valeur_input * poids_connexion # Contribuation en faisait input * poids
                
                # Ajout du biais
                somme_ponderee += self.biais[index_couche - 1][index_neuronne]

                # Ajout de la valeur pondérée avant activation
                weighted_couche.append(somme_ponderee)

                # On ajoute la fonction d'activation (pour les test, c'est du linéaire)
                activation = self.f(somme_ponderee)

                # On stock l'activation du neuronne
                activations_couche.append(activation)

            # Stocker les resultats des sommes pondérées de la couche
            weighted_values.append(weighted_couche)

            # Une fois la couche finie, les activations deviennent les inputs de la couche suivante
            activations = activations_couche
            activations_per_couche.append(activations_couche)

        return activations, weighted_values, activations_per_couche

    def loss(self, prediction, response):
        if isinstance(prediction, list):
            prediction = prediction[0]
        # Pour la backprop, on veut la dérivée de la perte, pas la perte elle-même !
        return 2 * (prediction - response)  # Dérivée de (prediction - response)²

    def backward_propagation(self, outputs, weighted_values, labels):
        """
        Calcul les deltas pour chaque neurone du réseau multi-couches.

        inputs :
            outputs         : activations de la couche de sortie
            weighted_values : valeurs pondérées de chaque neurone (avant activation)
            labels          : valeurs cibles

        return :
            deltas : liste de deltas par couche (couche d'entrée non incluse)
        """

        deltas_out = []
        for output in range(len(outputs)):
            loss_output = self.loss(outputs[output], labels[output])
            weighted_out = weighted_values[-1][output]
            delta_out = loss_output * self.df(weighted_out)
            deltas_out.append(delta_out)

        # On initialise la liste des deltas
        deltas = []
        deltas.insert(0, deltas_out)

        for couche in range(len(self.model_size) - 2, 0, -1):  # On ignore input et output
            delta_couche = []

            for neurone in range(self.model_size[couche]):
                suiv_poids = self.poids[couche]  # ← C'EST ICI LE PROBLÈME
                suiv_deltas = deltas[0]

                neuronne_delta = 0
                for delta in range(len(suiv_deltas)):
                    # AVANT : neuronne_delta += suiv_deltas[delta] * suiv_poids[delta]
                    # APRÈS : 
                    neuronne_delta += suiv_deltas[delta] * suiv_poids[delta][neurone]  # ← AJOUTE [neurone]

                neuronne_delta *= self.df(weighted_values[couche-1][neurone])  # ← ATTENTION : j'ai changé l'index ici !
                delta_couche.append(neuronne_delta)

            deltas.insert(0, delta_couche)

        self.deltas = deltas
        return deltas

    def update_weights_biases(self, activations, learning_rate=0.01):
        
        for couche in range(len(self.model_size) - 1):
            nb_neurones = self.model_size[couche + 1]
            nb_inputs = self.model_size[couche]

            for neurone in range(nb_neurones):
                for input in range(nb_inputs):
                    input_val = activations[couche][input]
                    
                    self.poids[couche][neurone][input] -= learning_rate * self.deltas[couche][neurone] * input_val
                
                self.biais[couche][neurone] -= learning_rate * self.deltas[couche][neurone]

# Exemple d'utilisation :
# model_size = [1, 10, 10, 1]
# nn = NeuralNetwork(model_size)
# outputs, weighted_values, activations_per_couche = nn.forward([0.5])
# deltas = nn.backward_propagation(outputs, weighted_values, [1.0])
# nn.update_weights_biases(activations_per_couche, 0.01)