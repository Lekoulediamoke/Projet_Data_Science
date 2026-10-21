# ==============================================================
# Calcul des notes des étudiants avec NumPy
# ==============================================================

# 1. Importer la bibliothèque NumPy
import numpy as np


def saisir_entier(message):
    """Redemande tant que l'utilisateur n'entre pas un entier positif."""
    while True:
        try:
            valeur = int(input(message))
            if valeur > 0:
                return valeur
            print("Le nombre doit être supérieur à 0.")
        except ValueError:
            print("Ce n'est pas un nombre entier valide. Réessayez.")


def saisir_note(message):
    """Redemande tant que la note n'est pas comprise entre 0 et 100."""
    while True:
        try:
            note = float(input(message))
            if 0 <= note <= 100:
                return note
            print("La note doit être comprise entre 0 et 100.")
        except ValueError:
            print("Ce n'est pas une note valide. Réessayez.")


def calculer_mention(pourcentage):
    """7. Renvoie la mention correspondant au pourcentage."""
    if pourcentage >= 90:
        return "A+"
    elif pourcentage >= 80:
        return "A"
    elif pourcentage >= 70:
        return "B+"
    elif pourcentage >= 60:
        return "B"
    elif pourcentage >= 50:
        return "C"
    else:
        return "F"


print("=== Calcul des notes des étudiants ===\n")

# 2. Nombre d'étudiants et de matières
nb_etudiants = saisir_entier("Entrez le nombre d'étudiants : ")
nb_matieres = saisir_entier("Entrez le nombre de matières : ")

# 3. Tableau NumPy de forme (nombre d'étudiants, nombre de matières)
notes = np.zeros((nb_etudiants, nb_matieres))
noms = []

# 4. Saisie du nom et des notes (sur 100) de chaque étudiant
for i in range(nb_etudiants):
    print(f"\n--- Étudiant {i + 1} ---")
    nom = input("Nom de l'étudiant : ").strip() or f"Étudiant {i + 1}"
    noms.append(nom)
    for j in range(nb_matieres):
        notes[i, j] = saisir_note(f"Note en matière {j + 1} (sur 100) : ")

# 5. Notes totales : somme de chaque ligne (axis=1 = par étudiant)
totaux = np.sum(notes, axis=1)

# 6. Pourcentage = (notes totales / (nombre de matières * 100)) * 100
pourcentages = (totaux / (nb_matieres * 100)) * 100

# 7. Mention de chaque étudiant
mentions = [calculer_mention(p) for p in pourcentages]

# 8. Affichage sous forme de tableau
print("\n" + "=" * 58)
print(f"{'Nom':<20}{'Total':>10}{'Pourcentage':>16}{'Mention':>12}")
print("=" * 58)
for i in range(nb_etudiants):
    print(f"{noms[i]:<20}{totaux[i]:>10.2f}{pourcentages[i]:>15.2f}%{mentions[i]:>12}")
print("=" * 58)
