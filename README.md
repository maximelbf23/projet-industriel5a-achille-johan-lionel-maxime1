# Analyse Thermique d'un Revêtement de Barrière Thermique (TBC)

Ce projet fournit un outil interactif pour l'analyse thermique d'un système de revêtement de barrière thermique (TBC) pour les aubes de turbine. L'application, développée en Python avec la bibliothèque Streamlit, permet aux utilisateurs d'explorer l'influence de divers paramètres sur le profil de température et d'évaluer les impacts associés en termes de masse, de coût et d'empreinte carbone.

## Contexte Scientifique

Le cœur de l'application est un modèle analytique de transfert de chaleur en régime permanent à travers une structure multicouche. Cette structure est composée de trois matériaux :

1.  **Superalliage** : Le matériau de base de l'aube.
2.  **Couche de liaison** : Assure l'adhésion entre le superalliage et la céramique.
3.  **Céramique (TBC)** : Une couche isolante qui protège le superalliage des hautes températures des gaz de combustion.

Le modèle résout l'équation de la chaleur en une dimension, mais prend en compte l'hétérogénéité et l'anisotropie du matériau, notamment dans la couche de céramique. Il calcule le profil de température et les flux de chaleur (normal et transverse) à travers l'épaisseur totale du matériau en fonction des conditions aux limites (températures imposées à la base et en surface) et des propriétés des matériaux.

## Structure du Code

Le projet est organisé en trois fichiers principaux :

-   `Profil de température Aube.py` : Le script principal qui exécute l'application web interactive avec Streamlit. Il gère l'interface utilisateur, les entrées, et la visualisation des résultats.
-   `core/calculation.py` : Le module de calcul. Il contient la logique pour résoudre le système d'équations thermiques (`solve_tbc_model`) et pour générer les profils de température et de flux (`calculate_profiles`).
-   `core/constants.py` : Un fichier qui centralise toutes les constantes physiques (conductivités thermiques, épaisseurs de référence), les conditions aux limites par défaut, et les paramètres pour l'analyse d'impact (densité, coût, etc.).

## Fonctionnalités de l'Application

L'interface se compose d'une barre latérale pour la configuration des paramètres et d'une zone principale avec trois onglets pour l'analyse des résultats.

### Barre Latérale : Paramètres

L'utilisateur peut ajuster les paramètres suivants :

-   **Épaisseur Céramique (α)** : Un facteur adimensionnel qui définit l'épaisseur de la couche de céramique par rapport à celle du superalliage.
-   **Anisotropie Céramique (β)** : Le rapport des conductivités thermiques dans la céramique, qui modélise comment la chaleur se propage préférentiellement dans une direction.
-   **Longueur d'Onde (Lw)** : Une taille caractéristique de défaut ou de variation spatiale de la température.
-   **Conditions aux Limites** : Températures à la base du superalliage et à la surface de la céramique.
-   **Scénario Catastrophe** : Permet de définir des conditions de température extrêmes pour calculer l'épaisseur de TBC nécessaire pour maintenir la température de l'alliage en dessous d'un seuil critique.

### Onglet 1 : Analyse Détaillée & Impacts

Cet onglet fournit une analyse complète pour une configuration unique :

-   **Indicateurs Clés (KPIs)** : Affiche l'épaisseur de la TBC en microns et la température calculée à l'interface critique entre le superalliage et la couche de liaison. Un statut visuel (✅, ⚠️, 🚨) indique si la température est dans une plage sûre.
-   **Graphiques des Profils** : Visualise les profils de température et de flux de chaleur (normal et transverse) à travers les trois couches du matériau. Des lignes horizontales indiquent les limites de température critiques et de sécurité.
-   **Tableau d'Impact** : Compare le cas nominal avec un "scénario catastrophe". Ce tableau quantifie l'impact de l'augmentation de l'épaisseur de la TBC nécessaire pour le scénario catastrophe en termes de surcharge de masse par aube, de coût et d'empreinte carbone.

### Onglet 2 : Étude Paramétrique (2D)

Cet onglet permet de simuler plusieurs valeurs du paramètre **α** (épaisseur de la céramique) et d'observer son influence sur :

-   La température à l'interface.
-   Le saut de flux transverse (un indicateur de l'hétérogénéité).

Les résultats sont présentés sous forme de graphiques et d'un tableau de synthèse détaillé qui inclut également les impacts (masse, coût, CO2) pour chaque valeur d'alpha testée.

### Onglet 3 : Cartographie 3D (Alpha/Beta)

Cet onglet offre une vue plus globale en générant une surface de réponse 3D. Il montre comment une variable (soit la température à l'interface, soit le saut de flux) évolue en fonction de la variation simultanée de **l'épaisseur (α)** et de **l'anisotropie (β)**. Cela permet d'identifier les zones de fonctionnement sûres et de comprendre les interactions complexes entre ces deux paramètres.

## Comment Lancer l'Application

1.  **Installation des dépendances :**
    Assurez-vous d'avoir Python installé. Ensuite, installez les bibliothèques nécessaires à partir du fichier `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

2.  **Lancement de l'application :**
    Exécutez la commande suivante dans votre terminal à la racine du projet.
    ```bash
    streamlit run "Profil de température Aube.py"
    ```

L'application s'ouvrira automatiquement dans votre navigateur web.
