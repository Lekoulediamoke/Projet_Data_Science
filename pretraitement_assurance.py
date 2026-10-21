# ==============================================================
# Nettoyage et mise à l'échelle d'un jeu de données d'assurance
# Entrée  : assurance_brut.csv   (créé par generer_donnees_assurance.py)
# Sortie  : assurance_propre.csv (prêt pour l'apprentissage automatique)
# ==============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", None)


# --------------------------------------------------------------
# 1. Charger le jeu de données
# --------------------------------------------------------------
df = pd.read_csv("assurance_brut.csv")
nb_lignes_brut = len(df)

print("=== 1. Données brutes ===")
print("Forme :", df.shape)
print(df.head(8))
print("\nTypes des colonnes (age, imc et frais devraient être numériques) :")
print(df.dtypes)


# --------------------------------------------------------------
# 2. Nettoyer les données
# --------------------------------------------------------------
print("\n=== 2. Diagnostic ===")
print("Valeurs manquantes par colonne :")
print(df.isna().sum())
print("\nValeurs différentes trouvées pour 'sexe'   :", sorted(df["sexe"].unique()))
print("Valeurs différentes trouvées pour 'fumeur' :", sorted(df["fumeur"].unique()))
print("Valeurs différentes trouvées pour 'region' :",
      sorted(df["region"].dropna().unique()))
print("Exemples de formats incohérents :")
print("  age   :", df.loc[df["age"].astype(str).str.contains("ans"), "age"].head(3).tolist())
print("  imc   :", df.loc[df["imc"].astype(str).str.contains(","), "imc"].head(3).tolist())
print("  frais :", df.loc[df["frais"].astype(str).str.contains("€"), "frais"].head(3).tolist())


# 2a. Corriger les formats de colonnes incohérents
print("\n=== 2a. Normalisation des formats ===")

# Colonnes numériques stockées en texte : on retire les unités et on
# remplace la virgule décimale par un point avant de convertir en nombre
df["age"] = pd.to_numeric(df["age"].astype(str).str.replace("ans", "").str.strip(),
                          errors="coerce")
df["imc"] = pd.to_numeric(df["imc"].astype(str).str.replace(",", "."), errors="coerce")
df["frais"] = pd.to_numeric(
    df["frais"].astype(str).str.replace("€", "").str.replace(",", "").str.strip(),
    errors="coerce")

# Colonnes texte : une seule écriture par valeur
df["sexe"] = df["sexe"].str.strip().str.lower().replace(
    {"h": "homme", "male": "homme", "f": "femme", "female": "femme"})
df["fumeur"] = df["fumeur"].astype(str).str.strip().str.lower().replace(
    {"yes": "oui", "1": "oui", "no": "non", "0": "non"})
df["region"] = (df["region"].str.strip().str.lower()
                .str.replace("_", "-").str.replace(" ", "-"))

print("sexe   :", sorted(df["sexe"].unique()))
print("fumeur :", sorted(df["fumeur"].unique()))
print("region :", sorted(df["region"].dropna().unique()))
print("Types après conversion :", dict(df.dtypes.astype(str)))


# 2b. Supprimer les doublons
# (après la normalisation, pour repérer aussi "Homme" / "homme", etc.)
nb_doublons = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
print(f"\n=== 2b. Doublons supprimés : {nb_doublons} ===")


# 2c. Traiter les valeurs aberrantes
# Règles métier : un assuré adulte a entre 18 et 100 ans,
# un IMC humain se situe entre 10 et 70. Au-delà : erreur de saisie.
print("\n=== 2c. Valeurs aberrantes ===")
aberrants_age = df[(df["age"] < 18) | (df["age"] > 100)]
aberrants_imc = df[(df["imc"] < 10) | (df["imc"] > 70)]
print("Âges impossibles :", aberrants_age["age"].tolist())
print("IMC impossibles  :", aberrants_imc["imc"].tolist())

# On les remplace par "manquant" : ils seront imputés à l'étape suivante,
# ce qui évite de perdre le reste de la ligne
df.loc[aberrants_age.index, "age"] = np.nan
df.loc[aberrants_imc.index, "imc"] = np.nan

# Vérification avec la règle de l'écart interquartile (IQR) sur les frais
q1, q3 = df["frais"].quantile([0.25, 0.75])
iqr = q3 - q1
hors_iqr = df[(df["frais"] < q1 - 1.5 * iqr) | (df["frais"] > q3 + 1.5 * iqr)]
print(f"Frais hors de l'intervalle IQR : {len(hors_iqr)} ligne(s) "
      "(conservées : des frais élevés sont plausibles, notamment chez les fumeurs)")


# 2d. Traiter les valeurs manquantes
print("\n=== 2d. Valeurs manquantes ===")
print("Avant traitement :", df.isna().sum().astype(int).to_dict())

# La cible 'frais' ne doit pas être inventée : on supprime ces lignes
nb_sans_frais = df["frais"].isna().sum()
df = df.dropna(subset=["frais"]).reset_index(drop=True)

# Numériques : médiane (peu sensible aux valeurs extrêmes)
for col in ["age", "imc", "enfants"]:
    df[col] = df[col].fillna(df[col].median())
# Catégorielle : valeur la plus fréquente
df["region"] = df["region"].fillna(df["region"].mode()[0])

df["age"] = df["age"].astype(int)
df["enfants"] = df["enfants"].astype(int)
print(f"Lignes sans frais supprimées : {nb_sans_frais}")
print("Après traitement :", df.isna().sum().astype(int).to_dict())

# Encodage des colonnes texte en nombres (nécessaire pour les modèles)
df["sexe"] = (df["sexe"] == "homme").astype(int)        # 1 = homme, 0 = femme
df["fumeur"] = (df["fumeur"] == "oui").astype(int)      # 1 = fumeur
df = pd.get_dummies(df, columns=["region"], prefix="region", dtype=int)

print("\nDonnées nettoyées :", df.shape)
print(df.head(8))


# --------------------------------------------------------------
# 3. Mise à l'échelle des caractéristiques
# --------------------------------------------------------------
print("\n=== 3. Mise à l'échelle ===")
colonnes_a_normaliser = ["age", "imc", "enfants"]

print("Statistiques AVANT :")
print(df[colonnes_a_normaliser].describe().loc[["mean", "std", "min", "max"]].round(2))

avant = df[colonnes_a_normaliser].head(8).copy()

scaler = StandardScaler()
df_normalise = df.copy()
df_normalise[colonnes_a_normaliser] = scaler.fit_transform(df[colonnes_a_normaliser])

print("\nAvant / après StandardScaler (5 premières lignes) :")
print(pd.concat([avant.head(), df_normalise[colonnes_a_normaliser].head().round(3)],
                axis=1, keys=["AVANT", "APRÈS"]))

print("\nStatistiques APRÈS (moyenne ~0, écart-type ~1) :")
print(df_normalise[colonnes_a_normaliser].describe()
      .loc[["mean", "std", "min", "max"]].round(2))

# Pour comparaison : ce que donnerait MinMaxScaler (valeurs entre 0 et 1)
minmax = MinMaxScaler().fit_transform(df[colonnes_a_normaliser])
print("\nPour comparaison, MinMaxScaler (5 premières lignes) :")
print(pd.DataFrame(minmax, columns=colonnes_a_normaliser).head().round(3))

df_normalise.to_csv("assurance_propre.csv", index=False)


# --------------------------------------------------------------
# Bilan
# --------------------------------------------------------------
print("\n=== Bilan ===")
print(f"Lignes : {nb_lignes_brut} au départ, {len(df_normalise)} à la fin")
print("Fichier prêt pour l'apprentissage : assurance_propre.csv")
print("Note : dans un vrai projet, on ajusterait le scaler sur les seules")
print("données d'entraînement, après train_test_split, pour éviter les fuites.")
