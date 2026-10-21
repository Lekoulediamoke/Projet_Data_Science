# ==============================================================
# Évaluation d'un modèle de classification
# Jeu de données : cancer du sein (Breast Cancer Wisconsin, sklearn)
# Modèle : régression logistique avec normalisation
# Stratégies : division entraînement/test puis validation croisée
# ==============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

pd.set_option("display.width", 120)


# --------------------------------------------------------------
# 1. Charger et préparer les données
# --------------------------------------------------------------
donnees = load_breast_cancer()

# Séparer les caractéristiques (X) et la cible (y)
X = pd.DataFrame(donnees.data, columns=donnees.feature_names)
y = donnees.target                         # 0 = malin, 1 = bénin
classes = ["malin", "bénin"]

print("=== 1. Chargement et préparation ===")
print("Forme des données :", X.shape, "(tumeurs x mesures)")
print("Répartition de la cible :",
      {classes[i]: int(n) for i, n in enumerate(np.bincount(y))})
print("Valeurs manquantes :", int(X.isna().sum().sum()))
print("\nAperçu de 5 caractéristiques :")
print(X.iloc[:5, :5])

# Prétraitement : les mesures ont des échelles très différentes
# (ex. aire ~ 1000, lissage ~ 0.1). On les normalise avec StandardScaler.
# Le pipeline applique la normalisation PUIS le modèle, et apprend la
# normalisation uniquement sur les données d'entraînement (pas de fuite).
print("\nÉchelles très différentes, d'où la normalisation :")
print(X[["mean area", "mean smoothness"]].describe().loc[["min", "max"]])


def creer_modele():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


# --------------------------------------------------------------
# 2. Diviser les données (80 % entraînement, 20 % test)
# --------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\n=== 2. Division des données ===")
print("Entraînement :", len(X_train), "tumeurs")
print("Test :", len(X_test), "tumeurs")


# --------------------------------------------------------------
# 3. Entraîner le modèle
# --------------------------------------------------------------
modele = creer_modele()
modele.fit(X_train, y_train)


# --------------------------------------------------------------
# 4. Évaluer sur l'ensemble de test
# --------------------------------------------------------------
y_pred = modele.predict(X_test)

print("\n=== 4. Évaluation sur l'ensemble de test ===")
precision_test = accuracy_score(y_test, y_pred)
print(f"Score de précision (accuracy) : {precision_test:.2%}")

matrice = confusion_matrix(y_test, y_pred)
print("\nMatrice de confusion :")
print(pd.DataFrame(matrice,
                   index=[f"vrai {c}" for c in classes],
                   columns=[f"prédit {c}" for c in classes]))

print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=classes))

plt.figure(figsize=(5, 4))
sns.heatmap(matrice, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")
plt.title("Matrice de confusion - régression logistique")
plt.tight_layout()
plt.savefig("matrice_confusion_cancer.png", dpi=120)
plt.show()

# Interprétation
vrais_malins, faux_benins = matrice[0]     # ligne des tumeurs réellement malignes
faux_malins, vrais_benins = matrice[1]     # ligne des tumeurs réellement bénignes
print("Interprétation :")
print(f"  - {precision_test:.1%} des tumeurs du test sont bien classées "
      f"({np.trace(matrice)}/{len(y_test)}).")
print(f"  - {faux_benins} tumeur(s) maligne(s) prédite(s) bénigne(s) : "
      "c'est l'erreur la plus grave (faux négatif), un cancer non détecté.")
print(f"  - {faux_malins} tumeur(s) bénigne(s) prédite(s) maligne(s) : "
      "fausse alerte, moins grave (des examens complémentaires corrigeraient).")
print(f"  - Rappel 'malin' = {vrais_malins}/{vrais_malins + faux_benins} : "
      "part des cancers réellement détectés. Dans un contexte médical, c'est")
print("    la mesure à surveiller en priorité, plus que l'accuracy seule.")


# --------------------------------------------------------------
# 5. Validation croisée à 5 plis sur le jeu de données complet
# --------------------------------------------------------------
print("\n=== 5. Validation croisée (5 plis) ===")
scores_cv = cross_val_score(creer_modele(), X, y, cv=5, scoring="accuracy")

for i, score in enumerate(scores_cv, start=1):
    print(f"  Pli {i} : {score:.2%}")
print(f"Moyenne : {scores_cv.mean():.2%}   Écart-type : {scores_cv.std():.2%}")

# Comparaison des deux stratégies
print("\n=== Comparaison ===")
comparaison = pd.DataFrame({
    "Stratégie": ["Division 80/20 (1 seul test)", "Validation croisée (5 plis)"],
    "Précision": [f"{precision_test:.2%}", f"{scores_cv.mean():.2%}"],
    "Tumeurs utilisées pour tester": [len(X_test), len(X)],
})
print(comparaison.to_string(index=False))

ecart = abs(precision_test - scores_cv.mean())
print(f"\nÉcart entre les deux estimations : {ecart:.2%}")
print("La division 80/20 dépend d'un seul découpage : avec un autre")
print("random_state, le score changerait un peu. La validation croisée teste")
print("le modèle 5 fois, chaque tumeur servant une fois au test : sa moyenne")
print("est plus fiable, et l'écart-type faible montre que le modèle est stable.")
if ecart < 0.02:
    print("Ici, les deux estimations sont proches : le modèle généralise bien")
    print("et le score obtenu sur le test n'est pas dû au hasard du découpage.")
else:
    print("Ici, l'écart est notable : le découpage 80/20 n'était pas")
    print("représentatif, il vaut mieux se fier à la validation croisée.")
