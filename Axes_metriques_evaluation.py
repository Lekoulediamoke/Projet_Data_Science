# ==============================================================
# Choisir et interpréter les métriques d'évaluation
# Partie 1 : classification  (cancer du sein, arbre de décision)
# Partie 2 : régression      (diabète, régression linéaire et arbre)
# Partie 3 : regroupement    (Iris sans étiquettes, K-Means)
# ==============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score,
                             silhouette_score, adjusted_rand_score)

pd.set_option("display.width", 120)


# ##############################################################
# PARTIE 1 - CLASSIFICATION
# ##############################################################
print("#" * 60)
print("PARTIE 1 - CLASSIFICATION")
print("#" * 60)

# 1. Données étiquetées : tumeurs du sein
cancer = load_breast_cancer()
X = cancer.data
y = 1 - cancer.target          # on inverse pour que 1 = malin (classe positive)
classes = ["bénin", "malin"]
print("Données :", X.shape[0], "tumeurs,", X.shape[1], "mesures")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 2. Classificateur : arbre de décision (profondeur limitée à 4)
arbre = DecisionTreeClassifier(max_depth=4, random_state=42)
arbre.fit(X_train, y_train)
y_pred = arbre.predict(X_test)

# 3. Évaluation
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rappel = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nMétriques (classe positive = malin) :")
print(f"  Exactitude (accuracy) : {acc:.2%}  -> part de prédictions justes")
print(f"  Précision             : {prec:.2%}  -> parmi les 'malin' prédits, part vraiment malins")
print(f"  Rappel                : {rappel:.2%}  -> parmi les vrais malins, part détectés")
print(f"  Score F1              : {f1:.2%}  -> équilibre entre précision et rappel")

matrice = confusion_matrix(y_test, y_pred)
vn, fp, fn, vp = matrice.ravel()
print("\nMatrice de confusion :")
print(pd.DataFrame(matrice,
                   index=[f"vrai {c}" for c in classes],
                   columns=[f"prédit {c}" for c in classes]))

plt.figure(figsize=(5, 4))
sns.heatmap(matrice, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")
plt.title("Classification - matrice de confusion")
plt.tight_layout()
plt.savefig("p1_matrice_confusion.png", dpi=120)
plt.show()

# 4. Interprétation
print("\nInterprétation :")
print(f"  - Vrais positifs  : {vp} cancers correctement détectés")
print(f"  - Vrais négatifs  : {vn} tumeurs bénignes correctement reconnues")
print(f"  - Faux négatifs   : {fn} cancers manqués (prédits bénins)")
print(f"  - Faux positifs   : {fp} fausses alertes (bénins prédits malins)")
print("  - Mesure la plus importante ici : le RAPPEL. Manquer un cancer")
print("    (faux négatif) est bien plus grave qu'une fausse alerte, qui sera")
print("    corrigée par des examens complémentaires. L'accuracy seule ne suffit")
print("    pas : elle traite les deux types d'erreur de la même façon.")


# ##############################################################
# PARTIE 2 - RÉGRESSION
# ##############################################################
print("\n" + "#" * 60)
print("PARTIE 2 - RÉGRESSION")
print("#" * 60)

# 1. Données : progression du diabète un an après (score de 25 à 346).
# Le jeu Boston Housing a été retiré de scikit-learn ; celui-ci est inclus.
diabete = load_diabetes()
X, y = diabete.data, diabete.target
print("Données :", X.shape[0], "patients,", X.shape[1], "mesures")
print(f"Cible : moyenne {y.mean():.0f}, écart-type {y.std():.0f}, "
      f"de {y.min():.0f} à {y.max():.0f}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


def evaluer_regression(nom, modele):
    """Entraîne le modèle et renvoie ses métriques (test et entraînement)."""
    modele.fit(X_train, y_train)
    pred_test = modele.predict(X_test)
    mse = mean_squared_error(y_test, pred_test)
    return {
        "Modèle": nom,
        "MAE": mean_absolute_error(y_test, pred_test),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R² test": r2_score(y_test, pred_test),
        "R² entraînement": r2_score(y_train, modele.predict(X_train)),
    }, pred_test


# 2-3. Trois modèles pour illustrer sous-adaptation et sur-adaptation
lineaire = LinearRegression()
res_lin, pred_lin = evaluer_regression("Régression linéaire", lineaire)
res_profond, _ = evaluer_regression("Arbre sans limite",
                                    DecisionTreeRegressor(random_state=42))
res_limite, _ = evaluer_regression("Arbre profondeur 3",
                                   DecisionTreeRegressor(max_depth=3, random_state=42))

resultats = pd.DataFrame([res_lin, res_profond, res_limite]).round(2)
print("\nMétriques :")
print(resultats.to_string(index=False))

plt.figure(figsize=(5, 5))
plt.scatter(y_test, pred_lin, alpha=0.7, edgecolor="k")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", label="Prédiction parfaite")
plt.xlabel("Valeur réelle")
plt.ylabel("Valeur prédite")
plt.title("Régression linéaire - réel vs prédit")
plt.legend()
plt.tight_layout()
plt.savefig("p2_reel_vs_predit.png", dpi=120)
plt.show()

# 4. Interprétation
print("\nInterprétation :")
print(f"  - MAE = {res_lin['MAE']:.1f} : en moyenne, la régression linéaire se")
print(f"    trompe d'environ {res_lin['MAE']:.0f} points sur un score allant de 25 à 346.")
print(f"  - RMSE = {res_lin['RMSE']:.1f} > MAE : la RMSE pénalise davantage les")
print("    grosses erreurs ; l'écart avec la MAE montre que quelques patients")
print("    sont très mal prédits. La MSE est dans une unité au carré,")
print("    donc peu parlante : on lit plutôt la RMSE.")
print(f"  - R² = {res_lin['R² test']:.2f} : le modèle explique environ "
      f"{res_lin['R² test']:.0%} des variations.")
print("    Le reste dépend de facteurs absents des données.")
print("  - Sous ou sur-adaptation ?")
print(f"    * Régression linéaire : R² entraînement {res_lin['R² entraînement']:.2f}"
      f" et test {res_lin['R² test']:.2f}, proches mais modestes")
print("      -> légère SOUS-ADAPTATION : le modèle est trop simple, mais il")
print("      généralise bien (pas de sur-adaptation).")
print(f"    * Arbre sans limite : R² entraînement {res_profond['R² entraînement']:.2f}"
      f" mais test {res_profond['R² test']:.2f}")
print("      -> SUR-ADAPTATION nette : il a appris les données par cœur.")
print(f"    * Limiter la profondeur à 3 réduit l'écart (test {res_limite['R² test']:.2f}),")
print("      mais la régression linéaire reste le meilleur modèle ici.")


# ##############################################################
# PARTIE 3 - REGROUPEMENT
# ##############################################################
print("\n" + "#" * 60)
print("PARTIE 3 - REGROUPEMENT")
print("#" * 60)

# 1. Données Iris SANS étiquettes (gardées à part pour l'ARI)
iris = load_iris()
X = iris.data
vraies_especes = iris.target

# Choix du nombre de groupes : on teste k de 2 à 6
print("Recherche du meilleur nombre de groupes :")
ks = range(2, 7)
inerties, silhouettes = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inerties.append(km.inertia_)
    silhouettes.append(silhouette_score(X, km.labels_))
    print(f"  k = {k} : inertie {km.inertia_:7.1f}   silhouette {silhouettes[-1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(list(ks), inerties, "o-")
axes[0].set_title("Méthode du coude (inertie)")
axes[0].set_xlabel("Nombre de groupes k")
axes[0].set_ylabel("Inertie")
axes[1].plot(list(ks), silhouettes, "o-", color="tab:orange")
axes[1].set_title("Score de silhouette")
axes[1].set_xlabel("Nombre de groupes k")
axes[1].set_ylabel("Silhouette")
for ax in axes:
    ax.set_xticks(list(ks))          # graduations entières
plt.tight_layout()
plt.savefig("p3_coude_silhouette.png", dpi=120)
plt.show()

# 2. K-Means avec k = 3
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
groupes = kmeans.fit_predict(X)

# 3. Évaluation
sil = silhouette_score(X, groupes)
inertie = kmeans.inertia_
ari = adjusted_rand_score(vraies_especes, groupes)
print("\nK-Means avec k = 3 :")
print(f"  Score de silhouette   : {sil:.3f}  (de -1 à 1, plus c'est haut mieux c'est)")
print(f"  Inertie               : {inertie:.1f}  (somme des distances au centre, plus bas = groupes compacts)")
print(f"  Indice Rand ajusté    : {ari:.3f}  (0 = hasard, 1 = identique aux vraies espèces)")

print("\nComparaison groupes trouvés / vraies espèces :")
print(pd.crosstab(pd.Series(vraies_especes, name="espèce").map(
                      dict(enumerate(iris.target_names))),
                  pd.Series(groupes, name="groupe")))

# Pistes d'amélioration testées
X_petales = X[:, 2:4]
g_petales = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X_petales)
g_gmm = GaussianMixture(n_components=3, random_state=42).fit(X).predict(X)
g_norm = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(
    StandardScaler().fit_transform(X))

pistes = pd.DataFrame({
    "Variante": ["K-Means, 4 mesures (référence)",
                 "K-Means, pétales seulement",
                 "Mélange gaussien (GMM)",
                 "K-Means, données normalisées"],
    "ARI": [ari,
            adjusted_rand_score(vraies_especes, g_petales),
            adjusted_rand_score(vraies_especes, g_gmm),
            adjusted_rand_score(vraies_especes, g_norm)],
}).round(3)
print("\nPistes d'amélioration testées :")
print(pistes.to_string(index=False))

# 4. Interprétation
print("\nInterprétation :")
print(f"  - Une silhouette de {sil:.2f} indique des groupes assez bien séparés.")
print(f"    Elle est maximale pour k = 2 ({max(silhouettes):.2f}) : versicolor et")
print("    virginica se chevauchent tellement qu'elles ressemblent à un seul groupe.")
print("  - Le tableau le confirme : setosa forme un groupe parfait, mais une partie")
print("    des virginica est mélangée aux versicolor.")
print(f"  - ARI = {ari:.2f} : bonne correspondance avec les vraies espèces, sans être parfaite.")
print("  - L'inertie baisse toujours quand k augmente : on cherche le 'coude'")
print("    de la courbe, ici autour de k = 3.")
print("  - Ce qui améliore les résultats :")
print("    * garder les mesures les plus utiles (pétales) : l'ARI monte nettement ;")
print("    * un modèle aux groupes ovales (mélange gaussien) au lieu de K-Means,")
print("      qui suppose des groupes ronds ;")
print("    * la normalisation n'aide pas ici : les mesures sont toutes en cm,")
print("      et elle réduit le poids des pétales, qui séparent le mieux les espèces.")
