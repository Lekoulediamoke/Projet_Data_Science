# ==============================================================
# Génération d'un jeu de données d'assurance synthétique "sale"
# (reproduit les problèmes courants des données réelles)
# Produit le fichier : assurance_brut.csv
# ==============================================================

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 200

age = rng.integers(18, 65, n)
imc = np.round(rng.normal(30, 5, n).clip(16, 50), 1)
enfants = rng.integers(0, 5, n)
sexe = rng.choice(["homme", "femme"], n)
fumeur = rng.choice(["oui", "non"], n, p=[0.2, 0.8])
region = rng.choice(["nord-est", "nord-ouest", "sud-est", "sud-ouest"], n)
frais = np.round(2000 + age * 250 + imc * 300 + enfants * 500
                 + np.where(fumeur == "oui", 20000, 0)
                 + rng.normal(0, 3000, n), 2)

df = pd.DataFrame({"age": age.astype(object), "imc": imc.astype(object),
                   "enfants": enfants.astype(float), "sexe": sexe,
                   "fumeur": fumeur, "region": region, "frais": frais.astype(object)})

# --- Formats incohérents ---
idx = rng.choice(n, 30, replace=False)
df.loc[idx[:10], "sexe"] = rng.choice(["H", "Homme", "male", "F", "Femme", "female"], 10)
df.loc[idx[10:20], "fumeur"] = rng.choice(["Oui", "NON", "yes", "no", "1", "0"], 10)
df.loc[idx[20:30], "region"] = rng.choice(["  Nord-Est", "SUD-OUEST ", "sud_est", "Nord Ouest"], 10)

idx = rng.choice(n, 25, replace=False)
df.loc[idx, "frais"] = [f"{float(v):,.2f} €" for v in df.loc[idx, "frais"]]      # "12,345.67 €"
idx = rng.choice(n, 15, replace=False)
df.loc[idx, "imc"] = [str(v).replace(".", ",") for v in df.loc[idx, "imc"]]     # "28,5"
idx = rng.choice(n, 10, replace=False)
df.loc[idx, "age"] = [f"{v} ans" for v in df.loc[idx, "age"]]                   # "45 ans"

# --- Valeurs aberrantes (erreurs de saisie) ---
df.loc[rng.choice(n, 3, replace=False), "age"] = [150, 5, 230]
df.loc[rng.choice(n, 2, replace=False), "imc"] = [95.0, 4.2]

# --- Valeurs manquantes ---
for col, nb in [("age", 8), ("imc", 12), ("enfants", 6), ("region", 4), ("frais", 3)]:
    df.loc[rng.choice(n, nb, replace=False), col] = np.nan

# --- Doublons ---
df = pd.concat([df, df.sample(12, random_state=1)], ignore_index=True)
df = df.sample(frac=1, random_state=2).reset_index(drop=True)

df.to_csv("assurance_brut.csv", index=False)
print("Fichier assurance_brut.csv créé :", df.shape[0], "lignes")
