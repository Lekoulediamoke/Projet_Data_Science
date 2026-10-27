# ==============================================================
# Identifier et corriger le surapprentissage dans un CNN
# Jeu de données : CIFAR-10 (images couleur 32 x 32, 10 classes)
#
# Modèle A : CNN sans régularisation -> on provoque le surapprentissage
# Modèle B : même CNN + augmentation + Batch Normalization + Dropout
# Modèle C : couches convolutionnelles de A GELÉES, seules les couches
#            supérieures sont réentraînées (avec augmentation et Dropout)
# ==============================================================

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report
from tensorflow import keras

keras.utils.set_random_seed(42)
CLASSES = ["avion", "voiture", "oiseau", "chat", "cerf",
           "chien", "grenouille", "cheval", "bateau", "camion"]
NB_TRAIN = 10000      # sous-ensemble : moins de données = surapprentissage plus visible
NB_VAL = 5000
EPOQUES = 20
TAILLE_LOT = 64


# --------------------------------------------------------------
# 1. Données
# --------------------------------------------------------------
(X, y), (X_test, y_test) = keras.datasets.cifar10.load_data()
y, y_test = y.ravel(), y_test.ravel()
X = X.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

melange = np.random.default_rng(42).permutation(len(X))
X_train, y_train = X[melange[:NB_TRAIN]], y[melange[:NB_TRAIN]]
X_val, y_val = X[melange[NB_TRAIN:NB_TRAIN + NB_VAL]], y[melange[NB_TRAIN:NB_TRAIN + NB_VAL]]
print(f"Entraînement : {len(X_train)} | Validation : {len(X_val)} | Test : {len(X_test)}")


# --------------------------------------------------------------
# 2. Construction des modèles
# --------------------------------------------------------------
augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),       # retournement : un chat reste un chat
    keras.layers.RandomRotation(0.05),           # rotation d'environ +/- 18 degrés
    keras.layers.RandomTranslation(0.1, 0.1),    # décalage = découpe d'une autre zone
], name="augmentation")


def blocs_convolutionnels(regularise):
    """3 blocs Conv -> (BatchNorm) -> ReLU -> MaxPool -> (Dropout)."""
    couches = []
    for filtres in (32, 64, 128):
        if regularise:
            couches += [keras.layers.Conv2D(filtres, 3, padding="same", use_bias=False),
                        keras.layers.BatchNormalization(),
                        keras.layers.Activation("relu"),
                        keras.layers.MaxPooling2D(),
                        keras.layers.Dropout(0.25)]
        else:
            couches += [keras.layers.Conv2D(filtres, 3, padding="same", activation="relu"),
                        keras.layers.MaxPooling2D()]
    return couches


def couches_superieures(regularise):
    couches = [keras.layers.Flatten(), keras.layers.Dense(256, activation="relu")]
    if regularise:
        couches.append(keras.layers.Dropout(0.5))
    couches.append(keras.layers.Dense(10, activation="softmax"))
    return couches


def compiler(couches, nom):
    modele = keras.Sequential([keras.Input(shape=(32, 32, 3))] + couches, name=nom)
    modele.compile(optimizer=keras.optimizers.Adam(1e-3),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modele


# --------------------------------------------------------------
# 3. Entraînement et évaluation
# --------------------------------------------------------------
historiques, resultats = {}, {}


def entrainer(modele, nom, epoques):
    print("\n" + "=" * 64)
    print("Modèle", nom)
    print("=" * 64)
    modele.summary()
    debut = time.time()
    h = modele.fit(X_train, y_train, epochs=epoques, batch_size=TAILLE_LOT,
                   validation_data=(X_val, y_val), verbose=2).history
    predictions = np.argmax(modele.predict(X_test, verbose=0), axis=1)
    historiques[nom] = h
    resultats[nom] = {
        "Précision entraînement": h["accuracy"][-1],
        "Précision validation": h["val_accuracy"][-1],
        "Écart entr. - valid.": h["accuracy"][-1] - h["val_accuracy"][-1],
        "Perte validation (min)": min(h["val_loss"]),
        "Perte validation (fin)": h["val_loss"][-1],
        "Précision TEST": np.mean(predictions == y_test),
        "Score F1 TEST (macro)": f1_score(y_test, predictions, average="macro"),
        "Durée (min)": (time.time() - debut) / 60,
    }
    return predictions


# Modèle A : on laisse le surapprentissage se produire
convs_A = blocs_convolutionnels(regularise=False)
modele_A = compiler(convs_A + couches_superieures(False), "A_sans_regularisation")
pred_A = entrainer(modele_A, "A : sans régularisation", EPOQUES)

# Signes de surapprentissage du modèle A
hA = historiques["A : sans régularisation"]
ep_min = int(np.argmin(hA["val_loss"])) + 1
print("\n=== Diagnostic du modèle A ===")
print(f"Précision entraînement : {hA['accuracy'][-1]:.2%}  (élevée ?)")
print(f"Précision validation   : {hA['val_accuracy'][-1]:.2%}  (bien plus faible ?)")
print(f"Perte de validation minimale à l'époque {ep_min}, "
      f"puis {min(hA['val_loss']):.3f} -> {hA['val_loss'][-1]:.3f} à la fin")
if hA["val_loss"][-1] > min(hA["val_loss"]) * 1.2:
    print("-> Les courbes divergent : la perte d'entraînement baisse pendant que")
    print("   celle de validation remonte. C'est du SURAPPRENTISSAGE.")

# Modèle B : augmentation + Batch Normalization + Dropout
modele_B = compiler([augmentation] + blocs_convolutionnels(True) + couches_superieures(True),
                    "B_regularise")
pred_B = entrainer(modele_B, "B : augmentation + BN + Dropout", EPOQUES)

# Modèle C : couches inférieures de A gelées, couches supérieures réentraînées
for couche in convs_A:
    couche.trainable = False          # ces poids ne bougeront plus
modele_C = compiler([augmentation] + convs_A + couches_superieures(True), "C_couches_gelees")
pred_C = entrainer(modele_C, "C : couches inférieures gelées", EPOQUES // 2)


# --------------------------------------------------------------
# 4. Comparaison
# --------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey="row")
for col, (nom, h) in enumerate(historiques.items()):
    ep = range(1, len(h["loss"]) + 1)
    axes[0, col].plot(ep, h["accuracy"], "o-", ms=3, label="Entraînement")
    axes[0, col].plot(ep, h["val_accuracy"], "o-", ms=3, label="Validation")
    axes[0, col].set_title(nom)
    axes[1, col].plot(ep, h["loss"], "o-", ms=3, label="Entraînement")
    axes[1, col].plot(ep, h["val_loss"], "o-", ms=3, label="Validation")
    axes[1, col].set_xlabel("Époque")
    for ligne in (0, 1):
        axes[ligne, col].grid(alpha=0.3)
        axes[ligne, col].legend()
axes[0, 0].set_ylabel("Précision")
axes[1, 0].set_ylabel("Perte")
plt.tight_layout()
plt.savefig("surapprentissage_courbes.png", dpi=120)

tableau = pd.DataFrame(resultats).T
print("\n=== Comparaison finale ===")
print(tableau.map(lambda v: f"{v:.4f}").to_string())

print("\n=== Rapport détaillé du modèle B (par classe) ===")
print(classification_report(y_test, pred_B, target_names=CLASSES, digits=3))

plt.show()
