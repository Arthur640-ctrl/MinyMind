# 📜 Historique des Modèles – MinyMind (Piéton)

Ce README liste les **différentes versions des modèles de réseau neuronal** utilisés pour les tests “piétons”.  
Chaque version contient un **tableau avec les résultats principaux**, et un lien vers le modèle dans le dossier `history/`.

---

## Table des versions

| Version | Description | Chemin | État |
|---------|-------------|--------|------|
| V1      | Modèle simple, supporte n'importe quelle taille de réseau, activation globale, forward/backward/weights update | `history/V1/model.py` | ✅ Fonctionnel |
| V2      | À compléter… | `history/V2/` | 🔜 À tester |
| V3      | À compléter… | `history/V3/` | 🔜 À tester |

---

## Version V1 – Résultats

**Modèle :** `model.py`  
**Description :**  
- Taille du réseau configurable `[entrées, cachés..., sortie]`  
- Initialisation des poids aléatoire ou à zéro  
- Fonction d'activation globale  
- Forward / Backward / Update des poids / Training

### 🔹 Fonction quadratique : `f(x) = x²`

| Itération | Erreur moyenne | x   | Valeur réelle | Prédiction | Erreur |
|-----------|----------------|-----|---------------|------------|--------|
| 1700      | 0.070741       | 0.25 | 0.062        | 0.326      | 0.264  |
| 1800      | 0.070108       | 0.55 | 0.303        | 0.335      | 0.033  |
| 1900      | 0.069216       | 0.75 | 0.562        | 0.341      | 0.221  |

### 🔹 Fonction sinus : `sin(x)`

| Itération | Erreur moyenne | x    | Valeur réelle | Prédiction | Erreur |
|-----------|----------------|------|---------------|------------|--------|
| 2700      | 0.001936       | 0.524 | 0.500       | 0.498      | 0.002  |
| 2800      | 0.001715       | 0.785 | 0.707       | 0.728      | 0.021  |
| 2900      | 0.001614       | 1.047 | 0.866       | 0.867      | 0.001  |
| -         | -              | 1.571 | 1.000       | 0.945      | 0.055  |

### 🔹 Problème XOR

| Input | Target | Prediction | Classe prédite |
|-------|--------|------------|----------------|
| [0,0] | 0      | 0.016      | 0              |
| [0,1] | 1      | 0.982      | 1              |
| [1,0] | 1      | 0.983      | 1              |
| [1,1] | 0      | 0.022      | 0              |

### 🔹 Régression linéaire : `f(x,y) = 2x + 3y`

| Itération | Erreur moyenne | Entrée  | Valeur réelle | Prédiction |
|-----------|----------------|---------|---------------|------------|
| 1700      | 0.000000       | (0.25,0.25) | 1.25   | 1.25       |
| 1800      | 0.000000       | (0.5,0.5)   | 2.50   | 2.50       |
| 1900      | 0.000000       | (0.75,0.25) | 2.25   | 2.25       |

### 🔹 Classification binaire

| x    | Classe réelle | Prediction | Classe prédite |
|------|---------------|------------|----------------|
| 0.20 | 0             | 0.001      | 0              |
| 0.45 | 0             | 0.243      | 0              |
| 0.55 | 1             | 0.775      | 1              |
| 0.80 | 1             | 0.998      | 1              |

---

## ⚡ Notes Version V1
- Modèle stable pour les fonctions simples  
- XOR et régression linéaire parfaitement résolus  
- Quadratique et sinus corrects, mais améliorable  

---

## 📂 Historique des modèles
- `history/V1/` → modèle simple, forward/backward/weights update  
- `history/V2/` → à compléter  
- `history/V3/` → à compléter  

