# ==============================================================
# Classification supervisée du jeu de données Iris
# Modèles : K plus proches voisins (KNN) et régression logistique
# ==============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.width", 120)        # affiche toutes les colonnes
pd.set_option("display.max_columns", None)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# --------------------------------------------------------------
# 1. Importer le jeu de données
# --------------------------------------------------------------
iris = load_iris()
X = iris.data                 # caractéristiques (150 fleurs x 4 mesures)
y = iris.target               # cible : 0 = setosa, 1 = versicolor, 2 = virginica
especes = [str(e) for e in iris.target_names]

df = pd.DataFrame(X, columns=iris.feature_names)
df["espece"] = [especes[i] for i in y]


# --------------------------------------------------------------
# 2. Explorer les données
# --------------------------------------------------------------
print("=== 2. Exploration des données ===")
print("Forme des données :", X.shape)
print("Noms des caractéristiques :", iris.feature_names)
print("Étiquettes des cibles :", especes)
print("\nPremières lignes :")
print(df.head())
print("\nNombre de fleurs par espèce :")
print(df["espece"].value_counts())


# --------------------------------------------------------------
# 3. Diviser les données (80 % entraînement, 20 % test)
# --------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\n=== 3. Division des données ===")
print("Entraînement :", X_train.shape[0], "fleurs")
print("Test :", X_test.shape[0], "fleurs")


# --------------------------------------------------------------
# 4. Choisir et entraîner un modèle : KNN
# --------------------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


# --------------------------------------------------------------
# 5. Faire des prédictions sur l'ensemble de test
# --------------------------------------------------------------
y_pred_knn = knn.predict(X_test)


# --------------------------------------------------------------
# 6. Évaluer les performances
# --------------------------------------------------------------
print("\n=== 6. Évaluation du modèle KNN ===")
precision_knn = accuracy_score(y_test, y_pred_knn)
print(f"Score de précision (accuracy) : {precision_knn:.2%}")

matrice = confusion_matrix(y_test, y_pred_knn)
print("\nMatrice de confusion :")
print(pd.DataFrame(matrice,
                   index=[f"vrai {e}" for e in especes],
                   columns=[f"prédit {e}" for e in especes]))

print("\nRapport détaillé :")
print(classification_report(y_test, y_pred_knn, target_names=especes))

# Visualisation de la matrice de confusion
plt.figure(figsize=(6, 5))
sns.heatmap(matrice, annot=True, fmt="d", cmap="Blues",
            xticklabels=especes, yticklabels=especes)
plt.xlabel("Espèce prédite")
plt.ylabel("Espèce réelle")
plt.title("Matrice de confusion - KNN (k = 5)")
plt.tight_layout()
plt.savefig("matrice_confusion_knn.png", dpi=120)
plt.show()

# Interprétation : taux de bonnes classifications pour chaque espèce
print("Interprétation, espèce par espèce :")
for i, espece in enumerate(especes):
    bien_classees = matrice[i, i]
    total = matrice[i].sum()
    print(f"  - {espece} : {bien_classees}/{total} fleurs bien classées "
          f"({bien_classees / total:.0%})")

erreurs = len(y_test) - np.trace(matrice)
if erreurs == 0:
    print("Le modèle classe parfaitement les 3 espèces sur l'ensemble de test.")
else:
    print(f"Le modèle commet {erreurs} erreur(s), situées hors de la diagonale.")
print("Setosa est très facile à séparer (pétales beaucoup plus petits).")
print("Versicolor et virginica se ressemblent davantage : c'est entre elles")
print("que les erreurs apparaissent en général.")
print(f"Attention : le test ne contient que {len(y_test)} fleurs, un score de")
print("100 % peut donc être un peu optimiste. La validation croisée ci-dessous")
print("donne une estimation plus fiable.")


# --------------------------------------------------------------
# Bonus 1 : deuxième modèle (régression logistique) et comparaison
# --------------------------------------------------------------
print("\n=== Bonus : comparaison avec la régression logistique ===")
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
precision_lr = accuracy_score(y_test, y_pred_lr)

print("Matrice de confusion - régression logistique :")
print(confusion_matrix(y_test, y_pred_lr))

# Validation croisée à 5 plis : le modèle est testé 5 fois sur des parties
# différentes des données, ce qui rend la comparaison plus solide
cv_knn = cross_val_score(KNeighborsClassifier(n_neighbors=5), X, y, cv=5)
cv_lr = cross_val_score(LogisticRegression(max_iter=200), X, y, cv=5)

comparaison = pd.DataFrame({
    "Modèle": ["KNN (k = 5)", "Régression logistique"],
    "Précision sur le test": [f"{precision_knn:.2%}", f"{precision_lr:.2%}"],
    "Validation croisée (moyenne)": [f"{cv_knn.mean():.2%}", f"{cv_lr.mean():.2%}"],
})
print("\n", comparaison.to_string(index=False))


# --------------------------------------------------------------
# Bonus 2 : limites de décision en 2D
# (avec seulement 2 caractéristiques : longueur et largeur du pétale)
# --------------------------------------------------------------
X_2d = X[:, 2:4]
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X_2d, y, test_size=0.2, random_state=42
)
knn_2d = KNeighborsClassifier(n_neighbors=5).fit(X2_train, y2_train)

# Grille de points couvrant tout le graphique
x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
zones = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(7, 5))
plt.contourf(xx, yy, zones, alpha=0.3, cmap="viridis")
nuage = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="viridis",
                    edgecolor="k", s=40)
plt.legend(nuage.legend_elements()[0], especes, title="Espèce")
plt.xlabel("Longueur du pétale (cm)")
plt.ylabel("Largeur du pétale (cm)")
plt.title("Limites de décision du KNN (2 caractéristiques)")
plt.tight_layout()
plt.savefig("limites_decision_knn.png", dpi=120)
plt.show()

print(f"\nPrécision du KNN avec seulement 2 caractéristiques : "
      f"{accuracy_score(y2_test, knn_2d.predict(X2_test)):.2%}")
