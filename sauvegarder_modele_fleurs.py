# ==============================================================
# Entraîne le classificateur de fleurs (MobileNetV2) et l'enregistre
# en .h5 pour l'application Streamlit.
# Amélioration par rapport à classificateur_fleurs.py : arrêt anticipé
# (early stopping) pour garder les poids de la meilleure époque.
# ==============================================================

import glob
import json
import os

import tensorflow as tf
from tensorflow import keras

keras.utils.set_random_seed(42)
TAILLE = 160
NOMS_FR = {"daisy": "marguerite", "dandelion": "pissenlit", "roses": "rose",
           "sunflowers": "tournesol", "tulips": "tulipe"}

# Données (déjà téléchargées lors de l'exercice précédent)
chemin = keras.utils.get_file(
    origin="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
    extract=True)
racine = chemin if os.path.isdir(chemin) else os.path.dirname(chemin)
dossier = os.path.dirname(glob.glob(os.path.join(racine, "**", "daisy"), recursive=True)[0])

entrainement, validation = keras.utils.image_dataset_from_directory(
    dossier, validation_split=0.2, subset="both", seed=42,
    image_size=(TAILLE, TAILLE), batch_size=32)
classes = [NOMS_FR.get(c, c) for c in entrainement.class_names]

pretraiter = keras.applications.mobilenet_v2.preprocess_input
entrainement = entrainement.map(lambda x, y: (pretraiter(x), y)).prefetch(tf.data.AUTOTUNE)
validation = validation.map(lambda x, y: (pretraiter(x), y)).prefetch(tf.data.AUTOTUNE)

# Même modèle que dans classificateur_fleurs.py
base = keras.applications.MobileNetV2(weights="imagenet", include_top=False,
                                      input_shape=(TAILLE, TAILLE, 3))
base.trainable = False
modele = keras.Sequential([
    keras.Input(shape=(TAILLE, TAILLE, 3)),
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
    base,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(len(classes), activation="softmax"),
], name="classificateur_fleurs")
modele.compile(optimizer=keras.optimizers.Adam(1e-3),
               loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Arrêt anticipé : on s'arrête si la perte de validation ne s'améliore plus
# pendant 3 époques, et on revient aux poids de la meilleure époque
arret = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                      restore_best_weights=True, verbose=1)
modele.fit(entrainement, validation_data=validation, epochs=15,
           callbacks=[arret], verbose=2)

perte, precision = modele.evaluate(validation, verbose=0)
print(f"\nModèle retenu : précision de validation {precision:.2%}, perte {perte:.4f}")

# Enregistrement du modèle et des noms de classes
modele.save("modele_fleurs.h5")
with open("classes_fleurs.json", "w", encoding="utf-8") as f:
    json.dump(classes, f, ensure_ascii=False)
print("Fichiers enregistrés : modele_fleurs.h5 et classes_fleurs.json")

# Vérification : le modèle rechargé prédit bien
recharge = keras.models.load_model("modele_fleurs.h5")
images, etiquettes = next(iter(validation))
predictions = recharge.predict(images[:5], verbose=0).argmax(axis=1)
print("Test du modèle rechargé (5 images) :")
for vraie, predite in zip(etiquettes[:5].numpy(), predictions):
    print(f"  réel : {classes[vraie]:<11} prédit : {classes[predite]}")
