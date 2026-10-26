# ==============================================================
# Point de contrôle 14 : augmentation des données et Dropout
# Jeu de données : Sign Language MNIST (ASL, 24 lettres, 28 x 28)
# On compare deux CNN identiques :
#   - modèle A : sans régularisation (comme au point de contrôle précédent)
#   - modèle B : avec augmentation des données + Dropout
# ==============================================================

import glob
import os
import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras

keras.utils.set_random_seed(42)
LETTRES = string.ascii_uppercase
EPOQUES = 10


# --------------------------------------------------------------
# Chargement des données (comme dans cnn_asl.py)
# --------------------------------------------------------------
def trouver_fichier(nom):
    trouves = glob.glob(os.path.join("**", nom), recursive=True)
    if trouves:
        return trouves[0]
    import kagglehub
    dossier = kagglehub.dataset_download("datamunge/sign-language-mnist")
    return glob.glob(os.path.join(dossier, "**", nom), recursive=True)[0]


def charger(chemin):
    df = pd.read_csv(chemin)
    y = df["label"].values
    X = df.drop(columns="label").values.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    return X, y


X, y = charger(trouver_fichier("sign_mnist_train.csv"))
X_test, y_test = charger(trouver_fichier("sign_mnist_test.csv"))
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
nb_classes = int(y.max()) + 1
print("Entraînement :", len(X_train), "| Validation :", len(X_val), "| Test :", len(X_test))


# --------------------------------------------------------------
# 1. Augmentation des données
# --------------------------------------------------------------
# Trois techniques, appliquées au hasard à chaque époque :
#   - rotation légère (environ +/- 15 degrés) : main un peu inclinée
#   - zoom (+/- 10 %) : main plus ou moins proche de la caméra
#   - décalage (+/- 10 %) : main pas parfaitement centrée
# Pas de retournement horizontal : il transformerait une main droite en
# main gauche et pourrait modifier la lettre signée.
augmentation = keras.Sequential([
    keras.layers.RandomRotation(0.04, fill_mode="nearest"),
    keras.layers.RandomZoom(0.1, fill_mode="nearest"),
    keras.layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"),
], name="augmentation")

# Affichage : chaque ligne = 1 image originale + 4 versions augmentées
exemples = X_train[:4]
fig, axes = plt.subplots(4, 5, figsize=(9, 8))
for ligne, image in enumerate(exemples):
    axes[ligne, 0].imshow(image.squeeze(), cmap="gray")
    axes[ligne, 0].set_title(f"Original ({LETTRES[y_train[ligne]]})")
    for col in range(1, 5):
        augmentee = np.array(augmentation(image[np.newaxis], training=True))[0]
        axes[ligne, col].imshow(augmentee.squeeze(), cmap="gray")
        axes[ligne, col].set_title(f"Augmentée {col}")
for ax in axes.ravel():
    ax.axis("off")
fig.suptitle("Images originales et augmentées")
plt.tight_layout()
plt.savefig("augmentation_exemples.png", dpi=120)


# --------------------------------------------------------------
# 2. Les deux modèles
# --------------------------------------------------------------
def creer_cnn(regularise):
    couches = [keras.Input(shape=(28, 28, 1))]
    if regularise:
        couches.append(augmentation)              # active seulement à l'entraînement
    couches += [
        keras.layers.Conv2D(32, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
    ]
    if regularise:
        couches.append(keras.layers.Dropout(0.25))  # éteint 25 % des sorties au hasard
    couches += [
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
    ]
    if regularise:
        couches.append(keras.layers.Dropout(0.25))
    couches += [
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
    ]
    if regularise:
        couches.append(keras.layers.Dropout(0.5))   # le plus fort, avant la sortie
    couches.append(keras.layers.Dense(nb_classes, activation="softmax"))

    modele = keras.Sequential(couches, name="B_regularise" if regularise else "A_simple")
    modele.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    return modele


resultats = {}
historiques = {}
for regularise in (False, True):
    modele = creer_cnn(regularise)
    nom = "B : augmentation + Dropout" if regularise else "A : sans régularisation"
    print("\n" + "=" * 60)
    print("Modèle", nom)
    print("=" * 60)
    modele.summary()
    h = modele.fit(X_train, y_train, epochs=EPOQUES, batch_size=64,
                   validation_data=(X_val, y_val), verbose=2).history
    _, acc_test = modele.evaluate(X_test, y_test, verbose=0)
    historiques[nom] = h
    resultats[nom] = {
        "Précision entraînement": h["accuracy"][-1],
        "Perte entraînement": h["loss"][-1],
        "Précision validation": h["val_accuracy"][-1],
        "Perte validation": h["val_loss"][-1],
        "Précision TEST": acc_test,
    }


# --------------------------------------------------------------
# 3. Comparaison
# --------------------------------------------------------------
tableau = pd.DataFrame(resultats).T
print("\n=== Comparaison après", EPOQUES, "époques ===")
print(tableau.map(lambda v: f"{v:.4f}").to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (nom, h) in zip(axes, historiques.items()):
    ep = range(1, len(h["loss"]) + 1)
    ax.plot(ep, h["accuracy"], "o-", label="Entraînement")
    ax.plot(ep, h["val_accuracy"], "o-", label="Validation")
    ax.axhline(resultats[nom]["Précision TEST"], color="red", ls="--", label="Test officiel")
    ax.set_title(nom)
    ax.set_xlabel("Époque")
    ax.set_ylabel("Précision")
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("regularisation_courbes.png", dpi=120)

a, b = resultats.values()
print("\n=== Analyse ===")
print(f"Précision sur le test : {a['Précision TEST']:.2%} sans régularisation, "
      f"{b['Précision TEST']:.2%} avec.")
gain = b["Précision TEST"] - a["Précision TEST"]
if gain > 0:
    print(f"La régularisation améliore la précision sur de nouvelles photos de {gain:.1%}.")
else:
    print("La régularisation n'améliore pas le test sur ces", EPOQUES, "époques.")
print(f"Précision d'entraînement : {a['Précision entraînement']:.2%} contre "
      f"{b['Précision entraînement']:.2%} : avec Dropout et augmentation, le modèle")
print("voit des images plus difficiles et ne peut plus compter sur chaque neurone,")
print("il apprend donc plus lentement mais mémorise moins.")

plt.show()
