# ==============================================================
# Classificateur d'images personnalisé par transfert d'apprentissage
# Jeu de données : photos de fleurs (5 classes, environ 3 700 photos)
# Modèle pré-entraîné : MobileNetV2 (ImageNet)
# Stratégie : EXTRACTION DE CARACTÉRISTIQUES (toutes les couches gelées)
# Tête : GlobalAveragePooling2D -> Dense -> Dropout -> Softmax
# ==============================================================

import glob
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow import keras

keras.utils.set_random_seed(42)
TAILLE = 160
TAILLE_LOT = 32
EPOQUES = 15
NOMS_FR = {"daisy": "marguerite", "dandelion": "pissenlit", "roses": "rose",
           "sunflowers": "tournesol", "tulips": "tulipe"}


# --------------------------------------------------------------
# 1. Jeu de données personnalisé : photos de fleurs
# --------------------------------------------------------------
# Archive publique de TensorFlow : un dossier par espèce de fleur
chemin = keras.utils.get_file(
    origin="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
    extract=True)
racine = chemin if os.path.isdir(chemin) else os.path.dirname(chemin)
dossier = os.path.dirname(glob.glob(os.path.join(racine, "**", "daisy"), recursive=True)[0])
print("Photos trouvées dans :", dossier)

# 80 % entraînement / 20 % validation, images redimensionnées en 160 x 160
entrainement, validation = keras.utils.image_dataset_from_directory(
    dossier, validation_split=0.2, subset="both", seed=42,
    image_size=(TAILLE, TAILLE), batch_size=TAILLE_LOT)
classes = [NOMS_FR.get(c, c) for c in entrainement.class_names]
print("Classes :", classes)

# Aperçu de quelques photos
plt.figure(figsize=(10, 4))
for images, etiquettes in entrainement.take(1):
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(classes[int(etiquettes[i])])
        plt.axis("off")
plt.suptitle("Exemples du jeu de données")
plt.tight_layout()
plt.savefig("fleurs_exemples.png", dpi=120)

# Prétraitement propre à MobileNetV2 : pixels ramenés entre -1 et 1
pretraiter = keras.applications.mobilenet_v2.preprocess_input
entrainement = entrainement.map(lambda x, y: (pretraiter(x), y)).prefetch(tf.data.AUTOTUNE)
validation = validation.map(lambda x, y: (pretraiter(x), y)).prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------------------
# 2. Charger MobileNetV2 pré-entraîné, sans son classificateur
# --------------------------------------------------------------
base = keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,                 # on supprime le classificateur à 1 000 classes
    input_shape=(TAILLE, TAILLE, 3))
base.trainable = False                 # extraction de caractéristiques : tout est gelé
print(f"\nMobileNetV2 : {len(base.layers)} couches, toutes gelées")


# --------------------------------------------------------------
# 3. Nouvelle tête de classification
# --------------------------------------------------------------
modele = keras.Sequential([
    keras.Input(shape=(TAILLE, TAILLE, 3)),
    # Augmentation légère pour limiter le surapprentissage
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
    base,                                         # extracteur de caractéristiques gelé
    keras.layers.GlobalAveragePooling2D(),        # 5 x 5 x 1280 -> vecteur de 1280
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(len(classes), activation="softmax"),   # 5 espèces de fleurs
], name="classificateur_fleurs")

modele.compile(optimizer=keras.optimizers.Adam(1e-3),
               loss="sparse_categorical_crossentropy", metrics=["accuracy"])
modele.summary()


# --------------------------------------------------------------
# 4. Entraînement
# --------------------------------------------------------------
print("\n=== Entraînement ===")
debut = time.time()
h = modele.fit(entrainement, validation_data=validation,
               epochs=EPOQUES, verbose=2).history
print(f"Durée : {(time.time() - debut) / 60:.1f} min")

# Courbes de précision et de perte
ep = range(1, EPOQUES + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(ep, h["accuracy"], "o-", label="Entraînement")
ax1.plot(ep, h["val_accuracy"], "o-", label="Validation")
ax1.set_title("Précision")
ax2.plot(ep, h["loss"], "o-", label="Entraînement")
ax2.plot(ep, h["val_loss"], "o-", label="Validation")
ax2.set_title("Perte")
for ax in (ax1, ax2):
    ax.set_xlabel("Époque")
    ax.grid(alpha=0.3)
    ax.legend()
plt.tight_layout()
plt.savefig("fleurs_courbes.png", dpi=120)


# --------------------------------------------------------------
# 5. Évaluation détaillée sur la validation
# --------------------------------------------------------------
# Prédictions lot par lot, pour garder chaque image avec sa vraie classe
vraies, predites = [], []
for images, etiquettes in validation:
    vraies.append(etiquettes.numpy())
    predites.append(np.argmax(modele.predict_on_batch(images), axis=1))
vraies, predites = np.concatenate(vraies), np.concatenate(predites)

print("\n=== Rapport par classe (validation) ===")
print(classification_report(vraies, predites, target_names=classes, digits=3))

matrice = confusion_matrix(vraies, predites)
plt.figure(figsize=(6, 5))
sns.heatmap(matrice, annot=True, fmt="d", cmap="Greens",
            xticklabels=classes, yticklabels=classes)
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.title("Matrice de confusion - validation")
plt.tight_layout()
plt.savefig("fleurs_matrice_confusion.png", dpi=120)


# --------------------------------------------------------------
# 6. Diagnostic : sous-apprentissage ou surapprentissage ?
# --------------------------------------------------------------
acc_train, acc_val = h["accuracy"][-1], h["val_accuracy"][-1]
ecart = acc_train - acc_val
meilleure = int(np.argmin(h["val_loss"])) + 1
remonte = h["val_loss"][-1] > min(h["val_loss"]) * 1.10

print("=== Diagnostic ===")
print(f"Précision finale : entraînement {acc_train:.2%}, validation {acc_val:.2%} "
      f"(écart {ecart:+.2%})")
print(f"Perte finale     : entraînement {h['loss'][-1]:.4f}, validation {h['val_loss'][-1]:.4f}")
print(f"Perte de validation minimale à l'époque {meilleure} ({min(h['val_loss']):.4f})")
print(f"Score F1 validation (macro) : {f1_score(vraies, predites, average='macro'):.4f}")
if acc_train < 0.80:
    print("-> SOUS-APPRENTISSAGE : précision d'entraînement encore faible.")
elif ecart > 0.05 or remonte:
    print("-> SURAPPRENTISSAGE : le modèle réussit nettement mieux sur l'entraînement,")
    print("   ou sa perte de validation remonte.")
else:
    print("-> BON ÉQUILIBRE : pas de sous-apprentissage ni de surapprentissage notable.")

plt.show()
