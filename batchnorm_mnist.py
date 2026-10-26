# ==============================================================
# Normalisation par lot (Batch Normalization) dans un CNN
# Jeu de données : MNIST (chiffres manuscrits 28 x 28, 10 classes)
# Comparaison :
#   a. CNN de référence, sans Batch Normalization
#   b. BN-CNN, avec une couche BatchNormalization dans chaque bloc
# Hyperparamètres identiques : SGD, taux 0.01, lots de 128, 20 époques
# ==============================================================

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras

EPOQUES = 20
TAILLE_LOT = 128
TAUX = 0.01
SEUIL = 0.98          # précision de validation visée pour mesurer la vitesse


# --------------------------------------------------------------
# 1. Données
# --------------------------------------------------------------
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
print("Entraînement :", X_train.shape, "| Test :", X_test.shape)


# --------------------------------------------------------------
# 2. Les deux modèles
# --------------------------------------------------------------
def bloc_convolutionnel(filtres, avec_bn):
    """Un bloc = convolution (+ Batch Normalization) + ReLU + max pooling."""
    if avec_bn:
        # BatchNorm placée entre la convolution et l'activation :
        # elle ramène les sorties de chaque filtre à une moyenne proche de 0
        # et un écart-type proche de 1, calculés sur le lot en cours.
        return [
            keras.layers.Conv2D(filtres, (3, 3), use_bias=False),  # biais inutile : BN le remplace
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.MaxPooling2D((2, 2)),
        ]
    return [
        keras.layers.Conv2D(filtres, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
    ]


def creer_cnn(avec_bn):
    keras.utils.set_random_seed(42)    # même point de départ aléatoire pour les deux
    couches = [keras.Input(shape=(28, 28, 1))]
    couches += bloc_convolutionnel(32, avec_bn)
    couches += bloc_convolutionnel(64, avec_bn)
    couches += [
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ]
    modele = keras.Sequential(couches, name="BN_CNN" if avec_bn else "CNN_reference")
    # Hyperparamètres strictement identiques pour les deux modèles
    modele.compile(optimizer=keras.optimizers.SGD(learning_rate=TAUX),
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    return modele


# --------------------------------------------------------------
# 3. Entraînement des deux modèles
# --------------------------------------------------------------
historiques = {}
resultats = {}
for avec_bn in (False, True):
    nom = "b. BN-CNN" if avec_bn else "a. Référence"
    modele = creer_cnn(avec_bn)
    print("\n" + "=" * 60)
    print("Modèle", nom)
    print("=" * 60)
    modele.summary()

    debut = time.time()
    h = modele.fit(X_train, y_train, epochs=EPOQUES, batch_size=TAILLE_LOT,
                   validation_split=0.1, verbose=2).history
    duree = time.time() - debut
    _, acc_test = modele.evaluate(X_test, y_test, verbose=0)

    atteint = [i + 1 for i, a in enumerate(h["val_accuracy"]) if a >= SEUIL]
    # Stabilité : variation moyenne de la perte de validation d'une époque à l'autre
    variations = np.abs(np.diff(h["val_loss"]))

    historiques[nom] = h
    resultats[nom] = {
        "Perte entraînement (ép. 1)": f"{h['loss'][0]:.4f}",
        "Précision validation (ép. 1)": f"{h['val_accuracy'][0]:.2%}",
        f"Époque où val >= {SEUIL:.0%}": atteint[0] if atteint else "jamais",
        "Perte entraînement finale": f"{h['loss'][-1]:.4f}",
        "Perte validation finale": f"{h['val_loss'][-1]:.4f}",
        "Précision validation finale": f"{h['val_accuracy'][-1]:.2%}",
        "Précision TEST": f"{acc_test:.2%}",
        "Variation moy. perte val": f"{variations.mean():.4f}",
        "Durée par époque (s)": f"{duree / EPOQUES:.1f}",
    }


# --------------------------------------------------------------
# 4. Courbes et comparaison
# --------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
for nom, h in historiques.items():
    ep = range(1, len(h["loss"]) + 1)
    ax1.plot(ep, h["loss"], "-", label=f"{nom} - entraînement")
    ax1.plot(ep, h["val_loss"], "--", label=f"{nom} - validation")
    ax2.plot(ep, h["accuracy"], "-", label=f"{nom} - entraînement")
    ax2.plot(ep, h["val_accuracy"], "--", label=f"{nom} - validation")
ax1.set_title("Perte")
ax2.set_title("Précision")
for ax in (ax1, ax2):
    ax.set_xlabel("Époque")
    ax.set_xticks(range(1, EPOQUES + 1, 2))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("batchnorm_courbes.png", dpi=120)

print("\n=== Comparaison après", EPOQUES, "époques ===")
print(pd.DataFrame(resultats).to_string())

ref, bn = historiques["a. Référence"], historiques["b. BN-CNN"]
print("\n=== Analyse ===")
print(f"Convergence : après 1 époque, perte d'entraînement {ref['loss'][0]:.3f} "
      f"sans BN contre {bn['loss'][0]:.3f} avec BN.")
print(f"Après {EPOQUES} époques : {ref['loss'][-1]:.4f} sans BN contre "
      f"{bn['loss'][-1]:.4f} avec BN.")
print("Stabilité : variation moyenne de la perte de validation entre deux époques :")
print(f"  {resultats['a. Référence']['Variation moy. perte val']} sans BN, "
      f"{resultats['b. BN-CNN']['Variation moy. perte val']} avec BN "
      "(plus c'est petit, plus la courbe est régulière).")

plt.show()