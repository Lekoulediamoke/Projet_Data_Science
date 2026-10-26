# ==============================================================
# CNN simple sur la langue des signes américaine (ASL)
# Jeu de données : Sign Language MNIST (version simplifiée de l'ASL)
#   - images 28 x 28 en niveaux de gris, une lettre par image
#   - 24 lettres (J et Z sont exclues car elles demandent un mouvement)
# Objectif : détecter surapprentissage / sous-apprentissage
# ==============================================================

import glob
import os
import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras

keras.utils.set_random_seed(42)      # résultats reproductibles
LETTRES = string.ascii_uppercase     # étiquette 0 = A, 1 = B, ... 24 = Y


# --------------------------------------------------------------
# 1. Charger et prétraiter les données
# --------------------------------------------------------------
def trouver_fichier(nom):
    """Cherche le fichier CSV dans le dossier courant, sinon le télécharge."""
    trouves = glob.glob(os.path.join("**", nom), recursive=True)
    if trouves:
        return trouves[0]
    try:
        import kagglehub
        print("Téléchargement du jeu de données depuis Kaggle...")
        dossier = kagglehub.dataset_download("datamunge/sign-language-mnist")
        trouves = glob.glob(os.path.join(dossier, "**", nom), recursive=True)
        if trouves:
            return trouves[0]
    except Exception as erreur:
        print("Téléchargement automatique impossible :", erreur)
    raise FileNotFoundError(
        f"{nom} introuvable. Téléchargez le jeu 'Sign Language MNIST' sur Kaggle "
        "et placez les fichiers CSV dans ce dossier.")


def charger(chemin):
    """Lit le CSV : 1 colonne 'label' + 784 colonnes de pixels (28 x 28)."""
    df = pd.read_csv(chemin)
    y = df["label"].values
    X = df.drop(columns="label").values
    # Redimensionnement : chaque ligne de 784 pixels devient une image 28 x 28
    # avec 1 canal (niveaux de gris), le format attendu par un CNN
    X = X.reshape(-1, 28, 28, 1).astype("float32")
    # Normalisation : pixels de 0-255 ramenés entre 0 et 1
    X = X / 255.0
    return X, y


X, y = charger(trouver_fichier("sign_mnist_train.csv"))
nb_classes = int(y.max()) + 1

# Division entraînement / validation (80 % / 20 %)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("=== 1. Données ===")
print("Images d'entraînement :", X_train.shape)
print("Images de validation  :", X_val.shape)
print("Lettres présentes     :", "".join(LETTRES[i] for i in np.unique(y)))
print(f"Pixels après normalisation : min {X.min():.1f}, max {X.max():.1f}")

# Aperçu de quelques images
fig, axes = plt.subplots(2, 6, figsize=(10, 4))
for ax, image, etiquette in zip(axes.ravel(), X_train, y_train):
    ax.imshow(image.squeeze(), cmap="gray")
    ax.set_title(LETTRES[etiquette])
    ax.axis("off")
fig.suptitle("Exemples d'images ASL")
plt.tight_layout()
plt.savefig("asl_exemples.png", dpi=120)


# --------------------------------------------------------------
# 2. Construire un CNN simple
# --------------------------------------------------------------
modele = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    keras.layers.Conv2D(32, (3, 3), activation="relu"),   # détecte bords et traits
    keras.layers.MaxPooling2D((2, 2)),                     # réduit la taille de moitié
    keras.layers.Conv2D(64, (3, 3), activation="relu"),   # détecte des formes de doigts
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),                                # image -> vecteur
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(nb_classes, activation="softmax"),  # une probabilité par lettre
])
modele.summary()

modele.compile(optimizer="adam",
               loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])


# --------------------------------------------------------------
# 3. Entraîner le modèle
# --------------------------------------------------------------
print("\n=== 3. Entraînement ===")
historique = modele.fit(X_train, y_train,
                        epochs=10,
                        batch_size=64,
                        validation_data=(X_val, y_val),
                        verbose=2)
h = historique.history
epoques = range(1, len(h["loss"]) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(epoques, h["loss"], "o-", label="Entraînement")
ax1.plot(epoques, h["val_loss"], "o-", label="Validation")
ax1.set_title("Perte")
ax1.set_xlabel("Époque")
ax1.legend()
ax1.grid(alpha=0.3)
ax2.plot(epoques, h["accuracy"], "o-", label="Entraînement")
ax2.plot(epoques, h["val_accuracy"], "o-", label="Validation")
ax2.set_title("Précision")
ax2.set_xlabel("Époque")
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("asl_courbes.png", dpi=120)


# --------------------------------------------------------------
# 4. Analyser les performances
# --------------------------------------------------------------
print("\n=== 4. Analyse ===")
acc_train, acc_val = h["accuracy"][-1], h["val_accuracy"][-1]
ecart = acc_train - acc_val
meilleure_epoque = int(np.argmin(h["val_loss"])) + 1
val_loss_remonte = h["val_loss"][-1] > min(h["val_loss"]) * 1.10

print(f"Précision finale : entraînement {acc_train:.2%}, validation {acc_val:.2%}")
print(f"Écart : {ecart:.2%}")
print(f"Perte finale     : entraînement {h['loss'][-1]:.4f}, "
      f"validation {h['val_loss'][-1]:.4f}")
print(f"Perte de validation la plus basse à l'époque {meilleure_epoque}")

print("\nQ1 - La précision d'entraînement est-elle beaucoup plus élevée ?")
if ecart > 0.05:
    print(f"  Oui : {ecart:.1%} d'écart, le modèle réussit mieux sur les images déjà vues.")
else:
    print(f"  Non : seulement {ecart:.1%} d'écart, les deux précisions sont proches.")

print("Q2 - La perte de validation augmente-t-elle pendant que l'autre baisse ?")
if val_loss_remonte:
    print(f"  Oui : elle est minimale à l'époque {meilleure_epoque} puis remonte.")
else:
    print("  Non : elle diminue ou reste stable avec la perte d'entraînement.")

print("\nDiagnostic :")
if acc_train < 0.80:
    print("  SOUS-APPRENTISSAGE : même sur l'entraînement, la précision reste faible.")
    print("  -> plus d'époques, plus de filtres ou un réseau plus profond.")
elif ecart > 0.05 or val_loss_remonte:
    print("  SURAPPRENTISSAGE : le modèle mémorise les images d'entraînement.")
    print("  -> Dropout, augmentation de données (rotations, zooms), arrêt anticipé.")
else:
    print("  BON FONCTIONNEMENT sur la validation : les deux courbes évoluent ensemble.")

# Vérification supplémentaire sur le jeu de test officiel (autres photos)
try:
    X_test, y_test = charger(trouver_fichier("sign_mnist_test.csv"))
    perte_test, acc_test = modele.evaluate(X_test, y_test, verbose=0)
    print(f"\nPrécision sur le jeu de TEST officiel : {acc_test:.2%}")
    if acc_val - acc_test > 0.05:
        print("  Nettement plus basse que la validation : les images d'entraînement et")
        print("  de validation proviennent des mêmes photos (versions légèrement")
        print("  modifiées), la validation est donc trop facile. Le test, fait")
        print("  d'autres photos, révèle un surapprentissage caché.")
except FileNotFoundError:
    print("\n(Jeu de test non trouvé : vérification supplémentaire ignorée.)")

plt.show()
