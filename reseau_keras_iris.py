# ==============================================================
# Réseau de neurones simple avec Keras (TensorFlow)
# Jeu de données : Iris (3 espèces de fleurs, 4 mesures)
# Architecture : 4 entrées -> 10 neurones ReLU -> 3 sorties softmax
# ==============================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from tensorflow import keras

keras.utils.set_random_seed(42)     # résultats reproductibles


# --------------------------------------------------------------
# 1. Données
# --------------------------------------------------------------
iris = load_iris()
X, y = iris.data, iris.target       # y : 0 = setosa, 1 = versicolor, 2 = virginica
especes = list(iris.target_names)

# Séparation entraînement / test (80 % / 20 %)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Normalisation : aide la descente de gradient à converger plus vite.
# Le scaler est ajusté sur l'entraînement uniquement (pas de fuite vers le test).
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Entraînement :", X_train.shape[0], "fleurs  |  Test :", X_test.shape[0], "fleurs")


# --------------------------------------------------------------
# 2. Création du réseau
# --------------------------------------------------------------
modele = keras.Sequential([
    keras.Input(shape=(4,), name="entree"),                          # couche d'entrée
    keras.layers.Dense(10, activation="relu", name="cachee"),        # couche cachée
    keras.layers.Dense(3, activation="softmax", name="sortie"),      # 3 probabilités
])

modele.summary()


# --------------------------------------------------------------
# 3. Compilation : fonction de perte + optimiseur
# --------------------------------------------------------------
# sparse_categorical_crossentropy : entropie croisée pour des classes
# codées 0, 1, 2. Adam ajuste automatiquement la taille des pas.
modele.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# --------------------------------------------------------------
# 4. Entraînement (la perte s'affiche à chaque époque)
# --------------------------------------------------------------
print("\n=== Entraînement ===")
historique = modele.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=2,
)

pertes = historique.history["loss"]
print(f"\nPerte d'entraînement : {pertes[0]:.4f} à l'époque 1 "
      f"-> {pertes[-1]:.4f} à l'époque {len(pertes)}")

plt.figure(figsize=(7, 4))
plt.plot(historique.history["loss"], label="Perte entraînement")
plt.plot(historique.history["val_loss"], label="Perte test")
plt.xlabel("Époque")
plt.ylabel("Entropie croisée")
plt.title("Évolution de la perte - réseau Keras sur Iris")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("courbe_perte_keras.png", dpi=120)
plt.show()


# --------------------------------------------------------------
# 5. Évaluation sur le test
# --------------------------------------------------------------
perte_test, precision_test = modele.evaluate(X_test, y_test, verbose=0)
print(f"\n=== Évaluation sur le test ===")
print(f"Perte : {perte_test:.4f}   Précision : {precision_test:.2%}")

predictions = np.argmax(modele.predict(X_test, verbose=0), axis=1)
print("Matrice de confusion (lignes = réel, colonnes = prédit) :")
print(confusion_matrix(y_test, predictions))

# Exemple de prédiction sur une nouvelle fleur
fleur = np.array([[5.1, 3.5, 1.4, 0.2]])       # mesures typiques d'une setosa
probas = modele.predict(scaler.transform(fleur), verbose=0)[0]
print("\nNouvelle fleur", fleur[0], ":")
for espece, p in zip(especes, probas):
    print(f"  {espece:<11} {p:.2%}")
print("Espèce prédite :", especes[np.argmax(probas)])


# --------------------------------------------------------------
# 6. Explication du processus d'apprentissage
print("\n=== Comment le réseau apprend ===")
print("1. Propagation avant : chaque neurone caché calcule une somme pondérée")
print("   de ses entrées puis applique ReLU ; la sortie softmax donne 3 probabilités.")
print("2. Erreur : l'entropie croisée compare ces probabilités à la vraie espèce.")
print("3. Rétropropagation : l'erreur remonte de la sortie vers l'entrée pour")
print("   calculer le gradient de chaque poids (sa part de responsabilité).")
print("4. Mise à jour : Adam déplace chaque poids dans le sens inverse de son")
print("   gradient, après chaque lot de 16 fleurs, pendant 50 époques.")
