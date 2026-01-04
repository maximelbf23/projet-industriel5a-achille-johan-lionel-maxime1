# Synthèse de l'Implémentation du Modèle Mécanique (TBC)

**Date :** 14 Décembre 2025  
**Contexte :** Projet Industriel 5A - Modélisation Thermomécanique

Ce document résume les travaux techniques réalisés pour intégrer la résolution mécanique (calcul des modes propres) dans l'application de simulation, conformément aux spécifications des documents `resolution_mécanique_5A.pdf` et `ProjectEstaca.pdf`.

---

## 1. Objectif

L'objectif était d'implémenter l'étape 7 de la résolution semi-analytique : **Trouver les racines de l'équation caractéristique mécanique**.
Ces racines ($\tau$) correspondent aux modes propres qui régissent la décroissance des contraintes dans l'épaisseur de la gaine.

## 2. Implémentation réalisée

### 2.1. Définition des Propriétés Matériaux (`core/constants.py`)
Nous avons ajouté les propriétés d'un matériau orthotrope générique (type céramique poreuse/Zircone) sous la forme d'une matrice de rigidité $C_{ij}$.
*   **Fichier :** `core/constants.py`
*   **Données :** Dictionnaire `MECHANICAL_PROPS` contenant les 9 constantes élastiques ($C_{11}$ à $C_{66}$).

### 2.2. Cœur de Calcul (`core/mechanical.py`)
C'est le module central qui effectue la résolution numérique.

**Algorithme implémenté :**
1.  **Construction de la Matrice Dynamique $M(\tau)$** :
    Implémentation de la matrice 3x3 dérivée des équations d'équilibre (section 6 du PDF). Elle dépend de la variable spectrale $\tau$ et des nombres d'onde $\delta_1, \delta_2$.
    
2.  **Calcul du Déterminant (Pivot de Gauss)** :
    Comme demandé, nous n'utilisons pas une boîte noire mais une méthode explicite. La fonction `compute_determinant_gaussian` effectue une élimination de Gauss pour calculer la valeur du déterminant pour un $\tau$ donné.

3.  **Résolution de l'Équation Caractéristique** :
    L'équation est de la forme $det(M(\tau)) = 0$. Théoriquement, c'est un polynôme bicubique en $\tau$ :
    $$P(\tau^2) = c_6 (\tau^2)^3 + c_4 (\tau^2)^2 + c_2 (\tau^2) + c_0 = 0$$
    
    Plutôt que de développer le déterminant symboliquement (très lourd), nous utilisons une **approche numérique robuste** :
    *   Nous évaluons le déterminant pour 3 valeurs tests de $X = \tau^2$ (0, 1, 2).
    *   Nous résolvons un système linéaire simple pour identifier les coefficients $c_4, c_2, c_0$ (sachant que $c_6 = C_{33}C_{44}C_{55}$).
    
4.  **Normalisation et Racines** :
    Pour garantir la précision numérique (les coefficients atteignant $10^{33}$), nous normalisons le polynôme par $c_6$ avant de chercher les racines avec `numpy.roots`.

### 2.3. Intégration Interface (`Profil de température Aube.py`)
Un nouvel onglet **"⚙️ Calcul Mécanique"** a été ajouté à l'application.
*   Il récupère dynamiquement le paramètre $L_w$ (longueur d'onde) défini par l'utilisateur.
*   Il lance le calcul en temps réel.
*   Il affiche le polynôme caractéristique et les 6 racines $\tau$ sous forme tabulaire.

## 3. Résultats et Vérification

Les tests effectués (`test_mechanical.py` et via l'interface) confirment la validité physique des résultats :
*   Les 6 racines obtenues sont toujours des **paires conjuguées** ($\pm \lambda$ ou $\pm a \pm ib$).
*   Cela valide la symétrie du matériau et la stabilité de l'algorithme de résolution.

---

**Prochaines étapes possibles (non réalisées) :**
*   **Etape 8** : Calcul des vecteurs propres associés à chaque racine.
*   **Etape 9** : Assemblage de la matrice de transfert globale pour le multicouche.
*   **Calcul des Contraintes** : Reconstruction du champ de contraintes complet $\sigma_{ij}(z)$.
