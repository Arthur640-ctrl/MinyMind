# 📜 Historique des Modèles – MinyMind

Ce README liste les **différentes versions des modèles de réseau neuronal** utilisés pour les tests.  
Chaque version contient un **tableau avec les résultats principaux**, et un lien vers le modèle dans le dossier `history/`.

---

## Table des versions

| Version | Description | Chemin | État |
|---------|-------------|--------|------|
| V1      | Modèle simple, supporte n'importe quelle taille de réseau, activation globale, forward/backward/weights update | `history/v1/model.py` | ✅ Fonctionnel |

---

## 📍 Version V1 – Résultats

**Modèle :** `model.py`  
**Description :**  
- Taille du réseau configurable `[entrées, cachés..., sortie]`  
- Initialisation des poids aléatoire ou à zéro  
- Fonction d'activation globale  
- Forward / Backward / Update des poids / Training

### 🏁 Résultats des tests :

| Test / Fonction           | Performance | Taille              | Epoques | Learning Rate | Activation | Loss Finale | Exemple                            | Conclusion                                                               |
| ------------------------- | ----------- | ------------------- | ------- | ------------- | ---------- | ----------- | ---------------------------------- | ------------------------------------------------------------------------ |
| XOR                       |⭐⭐⭐⭐⭐| \[2, 4, 1]          | 5000    | 0.5           | Sigmoid    | 0.000347    | - I : \[0, 0], T : 0, P : 0.016    | Parfait, maîtrise totale du XOR                                          |
| Quadratique x²            |⭐⭐⭐      | \[1, 20, 20, 1]     | 2000    | 0.1           | Sigmoid    | 0.069216    | - I : 0.25, T : 0.062, P : 0.326   | Réseau a du mal sur la courbure, améliorer avec plus de neurones/couches |
| Sin(x)                    |⭐⭐⭐⭐   | \[1, 30, 30, 30, 1] | 3000    | 0.05          | Tanh       | 0.001614    | - I : 0.524, T : 0.500, P : 0.498  | Très bon pour fonctions oscillatoires, légère erreur à π/2               |
| Régression multiple 2x+3y |⭐⭐⭐⭐⭐| \[2, 10, 1]         | 2000    | 0.01          | Linear     | 0.000000    | - I : \[0.5,0.5], T : 2.5, P : 2.5 | Parfait pour les fonctions linéaires multiples                           |
| Classification binaire    |⭐⭐⭐⭐   | \[1, 15, 1]         | 3000    | 0.1           | Sigmoid    | 0.002743    | - I : 0.55, T : 1, P : 0.775       | Très bon pour classification binaire, frontière nette                    |

---

## ⚡ Notes Version V1
- Modèle stable pour les fonctions simples  
- XOR et régression linéaire parfaitement maitrisés
- Quadratique et sinus corrects, mais améliorable dans le futur avec l'ajout de fonction d'activation spéciale par couche 

---

## 📂 Historique des modèles
- `history/V1/` → modèle simple, forward/backward/weights update

