# Rapport de prétraitement : données d'assurance

**Jeu de données** : `assurance_brut.csv`, 212 lignes et 7 colonnes (âge, IMC, enfants, sexe, fumeur, région, frais médicaux). Données synthétiques générées par `generer_donnees_assurance.py`, qui reproduit les défauts courants des données réelles.

**Scripts** : `pretraitement_assurance.py` produit `assurance_propre.csv` (197 lignes, prêt pour l'apprentissage).

## Points clés

1. **Problèmes trouvés dans les données brutes.** Les colonnes `age`, `imc` et `frais` étaient lues comme du texte à cause de formats mélangés (`"53 ans"`, `"28,1"` avec une virgule décimale, `"17,604.36 €"`). Les colonnes catégorielles avaient plusieurs écritures pour une même valeur : 7 pour le sexe (`H`, `Homme`, `male`…), 6 pour fumeur (`Oui`, `yes`, `1`…), 8 pour la région (majuscules, espaces, `sud_est`). On trouvait aussi 36 valeurs manquantes, 12 lignes en double et 5 valeurs impossibles (âges de 5, 150 et 230 ans, IMC de 4,2 et 95).

2. **Normalisation des formats, puis suppression des doublons.** Les unités et symboles ont été retirés et les colonnes converties en nombres. Chaque valeur catégorielle a été ramenée à une seule écriture (par exemple `homme`/`femme`, `oui`/`non`). Les doublons ont été supprimés **après** cette normalisation, pour repérer aussi les lignes identiques écrites différemment : 12 lignes retirées.

3. **Valeurs aberrantes et valeurs manquantes.** Les âges hors de 18 à 100 ans et les IMC hors de 10 à 70 ont été considérés comme des erreurs de saisie et remplacés par des valeurs manquantes, pour garder le reste de la ligne. Les 24 frais repérés par la règle de l'écart interquartile ont été conservés : des frais élevés sont plausibles, notamment chez les fumeurs. Les 3 lignes sans frais (la variable à prédire) ont été supprimées, car on ne doit pas inventer la cible. Les autres manques ont été remplacés par la **médiane** pour les nombres (peu sensible aux valeurs extrêmes) et par la **valeur la plus fréquente** pour la région.

4. **Encodage pour l'apprentissage.** `sexe` et `fumeur` ont été transformés en 0/1, et `region` en 4 colonnes binaires (encodage one-hot), car les modèles ne travaillent qu'avec des nombres.

5. **Mise à l'échelle avec StandardScaler, et pourquoi.** L'âge (18 à 64), l'IMC (17 à 45) et le nombre d'enfants (0 à 4) ont des échelles très différentes ; sans mise à l'échelle, l'âge dominerait les modèles basés sur les distances (KNN, K-Means) ou sur des coefficients (régression). J'ai choisi **StandardScaler** (moyenne 0, écart-type 1) plutôt que MinMaxScaler, car MinMaxScaler dépend entièrement du minimum et du maximum : une seule valeur extrême comprime toutes les autres. StandardScaler est moins sensible à ce problème et convient aux modèles linéaires et aux distances. Les colonnes 0/1 et la cible `frais` n'ont pas été mises à l'échelle.

## Avant / après la mise à l'échelle

| | âge | IMC | enfants |
|---|---|---|---|
| Moyenne avant | 41,31 | 29,60 | 2,09 |
| Écart-type avant | 12,48 | 4,66 | 1,37 |
| Moyenne après | 0,00 | 0,00 | 0,00 |
| Écart-type après | 1,00 | 1,00 | 1,00 |

*Remarque : dans un vrai projet, le scaler serait ajusté uniquement sur les données d'entraînement, après `train_test_split`, pour éviter toute fuite d'information vers le test.*
