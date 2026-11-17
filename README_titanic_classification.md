# Titanic – Prédiction de la survie des passagers

**Projet de classification en Python – Google Colab**  
*Travail réalisé en binôme dans le cadre de notre formation Data Analyst.*

## Présentation du projet

L'objectif de ce projet est de prédire si un passager du Titanic a survécu ou non à partir des informations dont nous disposons sur lui : son sexe, son âge, sa classe, le prix de son billet, etc.

Nous avons choisi de comparer trois algorithmes de classification : **Random Forest**, **régression logistique** et **KNN**. L'idée n'était pas seulement d'obtenir le meilleur score possible, mais aussi de comprendre l'effet du nettoyage des données, du *feature engineering* et du réglage des hyperparamètres sur les résultats.

Nous avons également cherché à répondre à une question plus descriptive : **quelles caractéristiques des passagers sont les plus associées à leur survie ?**

## Lancer le projet

Le projet a été développé et doit être présenté dans **Google Colab**. Le fichier Python `.py` est un export du notebook ; pour reproduire les analyses dans de bonnes conditions, nous recommandons d'utiliser directement Colab.

1. Ouvrir le [notebook Google Colab](https://colab.research.google.com/drive/1Z2DQxWHbeVz4_yyncsvoMlveFkUmBKrW) (un accès au fichier peut être nécessaire).
2. Se connecter à Google et démarrer une session d'exécution si Colab le demande.
3. Dans le menu **Exécution**, choisir **Tout exécuter**, ou lancer les cellules une par une dans l'ordre.
4. Consulter les tableaux, graphiques et commentaires au fil du notebook.

**Aucun fichier CSV n'est à importer manuellement :** les données sont chargées automatiquement depuis [notre fichier Titanic hébergé sur GitHub](https://raw.githubusercontent.com/dylanmalis-collab/titanic_csv/main/dataset-titanic-6afac2fa9ddd8043393546.csv). Une connexion Internet est donc nécessaire.

Les bibliothèques utilisées sont `pandas`, `numpy`, `matplotlib`, `seaborn` et `scikit-learn`. Elles sont généralement disponibles dans Colab. Les recherches d'hyperparamètres peuvent prendre un peu plus de temps que les autres cellules.

## Les données et notre démarche

Le jeu de données contient initialement **891 passagers**. La colonne à prédire est `Survived` : `0` pour un décès et `1` pour une survie.

Nous avons procédé par étapes :

1. **Explorer et nettoyer les données.** Nous avons notamment remplacé les âges manquants par la médiane et supprimé les deux lignes dont le port d'embarquement était inconnu. Nous avons écarté certaines colonnes peu exploitables directement, comme `PassengerId`, `Ticket` ou `Cabin`.
2. **Créer de nouvelles variables.** Nous avons testé plusieurs idées, puis conservé `Title` (titre extrait du nom), `IsChild` (moins de 16 ans) et `Sex_Pclass` (combinaison du sexe et de la classe). Nous avons ensuite comparé les modèles avec et sans ces nouvelles variables.
3. **Préparer les données pour chaque algorithme.** Les catégories sont encodées avec `OneHotEncoder`. Les variables numériques sont standardisées avec `StandardScaler` pour le KNN et la régression logistique, mais pas pour le Random Forest. Ces traitements sont regroupés dans des `Pipeline`.
4. **Entraîner et optimiser les modèles.** Les données sont séparées en 75 % pour l'entraînement et 25 % pour le test, avec `stratify` pour conserver la répartition des survivants et des victimes. Nous utilisons ensuite `RandomizedSearchCV` avec une validation croisée à 5 folds pour rechercher de meilleurs hyperparamètres.
5. **Comparer et interpréter les résultats.** Nous comparons les accuracies sur le train, en validation croisée et sur le test, avant de revenir aux caractéristiques des passagers à l'aide de matrices de corrélation.

## Résultats obtenus

Après optimisation des hyperparamètres, voici les résultats obtenus pour chaque modèle :

- **Random Forest** : 91,74 % sur le train, 83,49 % en validation croisée et 81,17 % sur le test.
- **Régression logistique** : 83,48 % sur le train, 82,13 % en validation croisée et 82,51 % sur le test.
- **KNN** : 84,38 % sur le train, 83,33 % en validation croisée et 80,27 % sur le test.

Nous avons retenu **la régression logistique** pour notre bilan final : elle obtient le meilleur score sur le jeu de test et ses résultats sont proches entre l'entraînement, la validation croisée et le test. Le Random Forest obtient toutefois la meilleure accuracy moyenne en validation croisée. Les différences restent modestes et ne suffisent pas à démontrer qu'un modèle serait toujours meilleur qu'un autre.

L'ajout des trois variables issues du *feature engineering* a notamment permis d'améliorer l'accuracy test de la régression logistique, passée de **78,03 % à 82,51 %** avant optimisation.

## Ce que nous apprend l'analyse des passagers

La dernière partie du notebook complète notre première matrice de corrélation avec les variables catégorielles, transformées en colonnes binaires grâce à `get_dummies()`.

Deux caractéristiques ressortent particulièrement :

- **Le sexe**, avec une corrélation de **+0,542** entre le fait d'être une femme et la survie (et **-0,542** pour les hommes).
- **La classe du passager**, avec une corrélation de **-0,336** pour `Pclass` : plus le numéro de classe augmente, plus la survie est faible dans notre échantillon.

La combinaison des deux est également parlante : être une femme de première classe est positivement associé à la survie, tandis qu'être un homme de troisième classe est négativement associé à celle-ci.

Il s'agit bien **d'associations statistiques** : une corrélation ne permet pas, à elle seule, de démontrer une relation de cause à effet.

## Limites et pistes d'amélioration

Ce travail reste un exercice d'apprentissage sur un jeu de données relativement petit. Les scores correspondent à un découpage précis des données et peuvent évoluer si celui-ci change. Une validation croisée répétée et l'étude d'autres métriques, comme le rappel ou le F1-score, permettraient d'aller plus loin.

Pour la présentation, le notebook sert de support principal : il permet de montrer le nettoyage, les prétraitements, les modèles et les résultats directement dans Colab. Les recherches d'hyperparamètres étant les cellules les plus longues, il est préférable d'avoir déjà exécuté le notebook avant la démonstration de 10 minutes.
