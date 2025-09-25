# Taille du modèle : nombre de neurones par couche
model_size = [2, 8, 35, 1]  # [input, couche cachée, output]

# Fonctions d'activations pour chaque couche (sauf input) 
model_activation = [0, "lin", "lin"]

# Listes pour stocker les poids et les biais
poids = []
biais = []

def f(x):
    return x

def init_network():
    # Génération des poids pour chaque couche (sauf la couche d'entrée)
    for couche_index in range(1, len(model_size)):
        # Nombre de neurones dans la couche actuelle
        nb_neurones_couche = model_size[couche_index]
        # Nombre de neurones dans la couche précédente (pour les connexions)
        nb_inputs_couche = model_size[couche_index - 1]

        # Liste pour stocker les poids de tous les neurones de cette couche
        poids_couche = []

        # Pour chaque neurone de la couche actuelle
        for neurone_index in range(nb_neurones_couche):
            # Liste pour stocker les poids venant de tous les neurones de la couche précédente
            poids_neurone = []

            # Pour chaque connexion venant de la couche précédente
            for connexion_index in range(nb_inputs_couche):
                poids_neurone.append(0)  # Initialisation du poids à 0

            # Ajouter les poids de ce neurone à la liste de la couche
            poids_couche.append(poids_neurone)

        # Ajouter tous les poids de cette couche à la liste globale des poids
        poids.append(poids_couche)

    print("Poids par couche :", poids)

    # Pour chaque couche du réseau, sauf la couche d'entrée
    for couche_index in range(1, len(model_size)):
        # Nombre de neurones dans la couche actuelle
        nb_neurones_couche = model_size[couche_index]

        # Liste pour stocker les biais de tous les neurones de cette couche
        biais_couche = []

        # Pour chaque neurone de la couche, on initialise son biais à 0
        for neurone_index in range(nb_neurones_couche):
            biais_couche.append(0)

        # Ajouter la liste des biais de cette couche à la liste globale des biais
        biais.append(biais_couche)

def forward(inputs):
    activations = inputs
    print("Input : ", inputs)

    # Calcul des activations couche par couches
    for index_couche in range(1, len(model_size)):

        # Liste qui va contenir les résultats (activations) de la couche actuelle
        activations_couche = []

        # Pour chaque neuronne dans la couche actuelle :
        for index_neuronne in range(model_size[index_couche]):
            
            # 1. On calcule la somme pndéerée (input * poids + biais)
            somme_ponderee = 0

            # On addition les resulatats de chaque neurone de la couche précédante (si c la couche input, on recup uniquement l'input)
            for index_input in range(model_size[index_couche - 1]):
                
                valeur_input = activations[index_input] # L'entrée
                poids_connexion = poids[index_couche - 1][index_neuronne][index_input] # Recuperaton du poids
                
                somme_ponderee += valeur_input * poids_connexion # Contribuation en faisait input * poids
            
            # Ajout du biais
            somme_ponderee += biais[index_couche - 1][index_neuronne]

            # On ajoute la fonction d'activation (pour les test, c'est du linéaire)
            activation = f(somme_ponderee)

            # On stock l'activation du neuronne
            activations_couche.append(activation)

        # Une fois la couche finie, les activations deviennent les inputs de la couche suivante
        activations = activations_couche

    return inputs

init_network()
output = forward([5.1, 0.45])
print("Output : ", output)