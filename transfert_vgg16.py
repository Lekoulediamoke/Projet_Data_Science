# ==============================================================
# Transfert d'apprentissage avec VGG16 (pré-entraîné sur ImageNet)
# Nouvelle tâche : classer les images de CIFAR-10 (10 classes)
#
# Phase 1 : base VGG16 gelée, on entraîne seulement la nouvelle tête
# Phase 2 : on débloque le dernier bloc de VGG16 (block5) et on affine
#           tout le modèle avec un taux d'apprentissage 100 fois plus faible
# ==============================================================

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report
from tensorflow import keras

VGG16 = keras.applications.VGG16
preprocess_input = keras.applications.vgg16.preprocess_input

keras.utils.set_random_seed(42)
CLASSES = ["avion", "voiture", "oiseau", "chat", "cerf",
           "chien", "grenouille", "cheval", "bateau", "camion"]
NB_TRAIN, NB_VAL, NB_TEST = 5000, 1000, 2000   # sous-ensemble : entraînement possible sur CPU
TAILLE = 64            # images agrandies de 32 x 32 à 64 x 64 pour VGG16
EPOQUES_PHASE1 = 5
EPOQUES_PHASE2 = 5
TAILLE_LOT = 32


# --------------------------------------------------------------
# 1. Jeu de données : CIFAR-10
# --------------------------------------------------------------
(X, y), (X_test, y_test) = keras.datasets.cifar10.load_data()
y, y_test = y.ravel(), y_test.ravel()

melange = np.random.default_rng(42).permutation(len(X))
X_train, y_train = X[melange[:NB_TRAIN]], y[melange[:NB_TRAIN]]
X_val, y_val = X[melange[NB_TRAIN:NB_TRAIN + NB_VAL]], y[melange[NB_TRAIN:NB_TRAIN + NB_VAL]]
X_test, y_test = X_test[:NB_TEST], y_test[:NB_TEST]

# Prétraitement propre à VGG16 : même transformation que lors de son
# entraînement sur ImageNet (canaux RVB -> BVR et soustraction des moyennes)
X_train = preprocess_input(X_train.astype("float32"))
X_val = preprocess_input(X_val.astype("float32"))
X_test = preprocess_input(X_test.astype("float32"))
print(f"Entraînement : {len(X_train)} | Validation : {len(X_val)} | Test : {len(X_test)}")


# --------------------------------------------------------------
# 2. Charger VGG16 pré-entraîné, sans ses couches de classification
# --------------------------------------------------------------
base = VGG16(weights="imagenet",          # poids appris sur 1,2 million d'images
             include_top=False,           # on retire les couches finales (1000 classes)
             input_shape=(TAILLE, TAILLE, 3))
base.trainable = False                    # on gèle tout l'extracteur de caractéristiques
nb_parametres = f"{base.count_params():,}".replace(",", " ")
print(f"\nVGG16 chargé : {len(base.layers)} couches, {nb_parametres} paramètres, tous gelés")


# --------------------------------------------------------------
# 3. Ajouter une tête de classification personnalisée
# --------------------------------------------------------------
modele = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),
    keras.layers.Resizing(TAILLE, TAILLE),           # 32 x 32 -> 64 x 64
    keras.layers.RandomFlip("horizontal"),           # petite augmentation
    base,                                            # extracteur VGG16
    keras.layers.Flatten(),                          # 2 x 2 x 512 -> 2048
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(len(CLASSES), activation="softmax"),   # 10 classes
], name="VGG16_transfert")


historiques = {}
resultats = {}


def entrainer(phase, epoques, taux):
    modele.compile(optimizer=keras.optimizers.Adam(learning_rate=taux),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print("\n" + "=" * 64)
    print(phase, f"(taux d'apprentissage {taux})")
    print("=" * 64)
    modele.summary()
    debut = time.time()
    h = modele.fit(X_train, y_train, epochs=epoques, batch_size=TAILLE_LOT,
                   validation_data=(X_val, y_val), verbose=2).history
    predictions = np.argmax(modele.predict(X_test, verbose=0), axis=1)
    historiques[phase] = h
    resultats[phase] = {
        "Paramètres entraînables": sum(int(np.prod(p.shape)) for p in modele.trainable_weights),
        "Précision entraînement": f"{h['accuracy'][-1]:.2%}",
        "Précision validation": f"{h['val_accuracy'][-1]:.2%}",
        "Précision TEST": f"{np.mean(predictions == y_test):.2%}",
        "Score F1 TEST (macro)": f"{f1_score(y_test, predictions, average='macro'):.4f}",
        "Durée (min)": f"{(time.time() - debut) / 60:.1f}",
    }
    return predictions


# --------------------------------------------------------------
# 4. Phase 1 : seule la nouvelle tête est entraînée
# --------------------------------------------------------------
entrainer("Phase 1 : base gelée", EPOQUES_PHASE1, 1e-3)


# --------------------------------------------------------------
# 5. Phase 2 : affinage (fine-tuning) du dernier bloc de VGG16
# --------------------------------------------------------------
# Les premiers blocs détectent des motifs généraux (bords, couleurs, textures)
# utiles pour toute image : on les laisse gelés. Le bloc 5 détecte des formes
# plus spécifiques à ImageNet : on l'adapte à CIFAR-10.
base.trainable = True
for couche in base.layers:
    couche.trainable = couche.name.startswith("block5")
debloquees = [c.name for c in base.layers if c.trainable]
print("\nCouches de VGG16 débloquées pour la phase 2 :", debloquees)

# Taux 100 fois plus faible : on ajuste finement les poids pré-entraînés
# sans détruire ce qu'ils ont appris
predictions = entrainer("Phase 2 : affinage du bloc 5", EPOQUES_PHASE2, 1e-5)


# --------------------------------------------------------------
# 6. Résultats
# --------------------------------------------------------------
print("\n=== Comparaison des deux phases ===")
print(pd.DataFrame(resultats).to_string())

print("\n=== Rapport détaillé après la phase 2 (par classe) ===")
print(classification_report(y_test, predictions, target_names=CLASSES, digits=3))

# Courbes des deux phases mises bout à bout
h1, h2 = historiques.values()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, cle, titre in ((ax1, "accuracy", "Précision"), (ax2, "loss", "Perte")):
    entr = h1[cle] + h2[cle]
    val = h1["val_" + cle] + h2["val_" + cle]
    ep = range(1, len(entr) + 1)
    ax.plot(ep, entr, "o-", label="Entraînement")
    ax.plot(ep, val, "o-", label="Validation")
    ax.axvline(len(h1[cle]) + 0.5, color="gray", ls="--")
    ax.text(len(h1[cle]) + 0.6, ax.get_ylim()[1], " début phase 2", va="top", color="gray")
    ax.set_title(titre)
    ax.set_xlabel("Époque")
    ax.grid(alpha=0.3)
    ax.legend()
plt.tight_layout()
plt.savefig("transfert_vgg16_courbes.png", dpi=120)
plt.show()
