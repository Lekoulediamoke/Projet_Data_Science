# ==============================================================
# Affinage (fine-tuning) de BERT pour l'analyse de sentiments
# Modèle : bert-base-uncased (Hugging Face)
# Jeu de données : SST-2 (critiques de films en anglais, positif / négatif)
# Entraînement avec Trainer, suivi de la précision et de la perte
# ==============================================================

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report)
from transformers import (BertTokenizer, BertForSequenceClassification,
                          Trainer, TrainingArguments, TrainerCallback, set_seed)

set_seed(42)
NB_ENTRAINEMENT = 3000      # sous-ensemble : l'affinage reste faisable sur CPU
LONGUEUR_MAX = 64           # les phrases de SST-2 sont courtes
EPOQUES = 3
TAILLE_LOT = 16
TAUX = 2e-5
ETIQUETTES = ["négatif", "positif"]


# --------------------------------------------------------------
# 1. Jeu de données : SST-2
# --------------------------------------------------------------
sst2 = load_dataset("nyu-mll/glue", "sst2")
entrainement = sst2["train"].shuffle(seed=42).select(range(NB_ENTRAINEMENT))
validation = sst2["validation"]          # 872 phrases annotées
print(f"Entraînement : {len(entrainement)} phrases | Validation : {len(validation)} phrases")
print("Exemples :")
for exemple in entrainement.select(range(3)):
    print(f"  [{ETIQUETTES[exemple['label']]}] {exemple['sentence']}")


# --------------------------------------------------------------
# 2. Tokenizer et modèle BERT pré-entraîné
# --------------------------------------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
modele = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,                                   # nouvelle tête : 2 classes
    id2label={0: "négatif", 1: "positif"},
    label2id={"négatif": 0, "positif": 1})
nb_params = sum(p.numel() for p in modele.parameters())
print(f"\nBERT chargé : {nb_params / 1e6:.0f} millions de paramètres")

# Exemple de tokenisation
phrase = "This movie was surprisingly good!"
print(f"\nTokenisation de « {phrase} » :")
print("  Jetons :", tokenizer.tokenize(phrase))
print("  Identifiants :", tokenizer(phrase)["input_ids"])


def tokeniser(lot):
    # Découpe en jetons, ajoute [CLS] et [SEP], complète ou tronque à 64 jetons
    return tokenizer(lot["sentence"], padding="max_length",
                     truncation=True, max_length=LONGUEUR_MAX)


entrainement = entrainement.map(tokeniser, batched=True)
validation = validation.map(tokeniser, batched=True)
colonnes = ["input_ids", "attention_mask", "token_type_ids", "label"]
entrainement.set_format("torch", columns=colonnes)
validation.set_format("torch", columns=colonnes)
# 1 000 phrases d'entraînement pour mesurer la précision d'entraînement
entrainement_mesure = entrainement.select(range(1000))


# --------------------------------------------------------------
# 3. Affinage avec Trainer
# --------------------------------------------------------------
def calculer_metriques(prediction):
    predites = np.argmax(prediction.predictions, axis=1)
    return {"accuracy": accuracy_score(prediction.label_ids, predites),
            "f1": f1_score(prediction.label_ids, predites)}


class SuiviEpoques(TrainerCallback):
    """Mesure précision et perte (entraînement et validation) à chaque époque."""
    def __init__(self):
        self.trainer = None
        self.historique = []

    def mesurer(self, epoque):
        entr = self.trainer.evaluate(entrainement_mesure, metric_key_prefix="entr")
        val = self.trainer.evaluate(validation, metric_key_prefix="val")
        self.historique.append({
            "époque": epoque,
            "perte entraînement": entr["entr_loss"],
            "précision entraînement": entr["entr_accuracy"],
            "perte validation": val["val_loss"],
            "précision validation": val["val_accuracy"],
            "F1 validation": val["val_f1"],
        })
        print(f"\n>>> Époque {epoque} : perte entr. {entr['entr_loss']:.4f}, "
              f"précision entr. {entr['entr_accuracy']:.2%} | perte val. "
              f"{val['val_loss']:.4f}, précision val. {val['val_accuracy']:.2%}\n")

    def on_epoch_end(self, args, state, control, **kwargs):
        self.mesurer(round(state.epoch))


arguments = TrainingArguments(
    output_dir="bert_sortie",
    per_device_train_batch_size=TAILLE_LOT,
    per_device_eval_batch_size=64,
    learning_rate=TAUX,                 # petit taux : on ajuste BERT sans l'abîmer
    num_train_epochs=EPOQUES,
    weight_decay=0.01,
    eval_strategy="no",                 # les évaluations sont faites par SuiviEpoques
    save_strategy="no",                 # on n'enregistre pas le modèle (440 Mo)
    logging_steps=50,
    report_to="none",
    seed=42,
)

suivi = SuiviEpoques()
trainer = Trainer(model=modele, args=arguments, train_dataset=entrainement,
                  compute_metrics=calculer_metriques, callbacks=[suivi])
suivi.trainer = trainer

print("\n=== Avant affinage ===")
suivi.mesurer(0)                        # BERT n'a pas encore appris la tâche

print("=== Affinage ===")
debut = time.time()
trainer.train()
print(f"Durée de l'affinage : {(time.time() - debut) / 60:.1f} min")


# --------------------------------------------------------------
# 4. Courbes d'entraînement et de validation
# --------------------------------------------------------------
historique = pd.DataFrame(suivi.historique).set_index("époque")
print("\n=== Suivi par époque ===")
print(historique.round(4).to_string())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(historique.index, historique["précision entraînement"], "o-", label="Entraînement")
ax1.plot(historique.index, historique["précision validation"], "o-", label="Validation")
ax1.set_title("Précision")
ax2.plot(historique.index, historique["perte entraînement"], "o-", label="Entraînement")
ax2.plot(historique.index, historique["perte validation"], "o-", label="Validation")
ax2.set_title("Perte")
for ax in (ax1, ax2):
    ax.set_xlabel("Époque (0 = avant affinage)")
    ax.set_xticks(historique.index)
    ax.grid(alpha=0.3)
    ax.legend()
plt.tight_layout()
plt.savefig("bert_courbes.png", dpi=120)


# --------------------------------------------------------------
# 5. Évaluation finale sur la validation
# --------------------------------------------------------------
resultat = trainer.predict(validation)
predites = np.argmax(resultat.predictions, axis=1)
vraies = resultat.label_ids

print("\n=== Évaluation finale (validation) ===")
print(f"Précision : {accuracy_score(vraies, predites):.2%}")
print(f"Score F1  : {f1_score(vraies, predites):.4f}")
print(classification_report(vraies, predites, target_names=ETIQUETTES, digits=3))

matrice = confusion_matrix(vraies, predites)
print("Matrice de confusion (lignes = réel, colonnes = prédit) :")
print(pd.DataFrame(matrice, index=ETIQUETTES, columns=ETIQUETTES))
plt.figure(figsize=(5, 4))
sns.heatmap(matrice, annot=True, fmt="d", cmap="Blues",
            xticklabels=ETIQUETTES, yticklabels=ETIQUETTES)
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.title("BERT affiné - matrice de confusion")
plt.tight_layout()
plt.savefig("bert_matrice_confusion.png", dpi=120)


# --------------------------------------------------------------
# 6. Test sur mes propres phrases
# --------------------------------------------------------------
mes_phrases = [
    "I absolutely loved this film, the acting was brilliant.",
    "What a waste of time, the plot made no sense.",
    "The movie was not bad at all.",
    "I expected a masterpiece, but it was just boring.",
    "The soundtrack is great but the story is weak.",
]
modele.eval()
entrees = tokenizer(mes_phrases, padding=True, truncation=True,
                    max_length=LONGUEUR_MAX, return_tensors="pt").to(modele.device)
with torch.no_grad():
    probabilites = torch.softmax(modele(**entrees).logits, dim=1).cpu().numpy()

print("\n=== Test sur mes propres phrases ===")
for texte, proba in zip(mes_phrases, probabilites):
    classe = int(np.argmax(proba))
    print(f"[{ETIQUETTES[classe]:>7} {proba[classe]:.1%}] {texte}")

plt.show()