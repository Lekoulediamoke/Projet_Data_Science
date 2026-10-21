# ==============================================================
# Réseau de neurones de base, codé à la main avec NumPy
# Problème : XOR (classification binaire)
# Architecture : 2 entrées -> 4 neurones cachés -> 1 sortie
# ==============================================================

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)          # pour obtenir les mêmes résultats à chaque exécution
np.set_printoptions(precision=4, suppress=True)


# --------------------------------------------------------------
# Jeu de données XOR : la sortie vaut 1 si les deux entrées sont différentes
# --------------------------------------------------------------
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0],
              [1],
              [1],
              [0]])


# --------------------------------------------------------------
# Fonctions d'activation et leurs dérivées
# --------------------------------------------------------------
def tanh(z):
    return np.tanh(z)

def tanh_derivee(a):            # a = tanh(z)
    return 1 - a ** 2

def sigmoide(z):
    return 1 / (1 + np.exp(-z))  # renvoie une probabilité entre 0 et 1


# --------------------------------------------------------------
# Initialisation des poids et des biais (valeurs aléatoires)
# --------------------------------------------------------------
n_entrees, n_caches, n_sorties = 2, 4, 1

W1 = np.random.randn(n_entrees, n_caches)   # poids entrée -> couche cachée
b1 = np.zeros((1, n_caches))                # biais de la couche cachée
W2 = np.random.randn(n_caches, n_sorties)   # poids couche cachée -> sortie
b2 = np.zeros((1, n_sorties))               # biais de la sortie

taux_apprentissage = 0.5
nb_epoques = 3000


# --------------------------------------------------------------
# Structure du modèle
# --------------------------------------------------------------
print("=" * 58)
print("STRUCTURE DU MODÈLE")
print("=" * 58)
print(f"{'Couche':<22}{'Neurones':>9}{'Activation':>13}{'Paramètres':>13}")
print("-" * 58)
print(f"{'Entrée':<22}{n_entrees:>9}{'-':>13}{0:>13}")
print(f"{'Cachée':<22}{n_caches:>9}{'tanh':>13}{W1.size + b1.size:>13}")
print(f"{'Sortie':<22}{n_sorties:>9}{'sigmoïde':>13}{W2.size + b2.size:>13}")
print("-" * 58)
print(f"Total des paramètres entraînables : {W1.size + b1.size + W2.size + b2.size}")
print(f"Fonction de perte : entropie croisée binaire")
print(f"Taux d'apprentissage : {taux_apprentissage}   Époques : {nb_epoques}")


def passage_avant(X):
    """Forward pass : calcule la sortie du réseau à partir des entrées."""
    z1 = X @ W1 + b1            # somme pondérée de la couche cachée
    a1 = tanh(z1)               # activation de la couche cachée
    z2 = a1 @ W2 + b2           # somme pondérée de la sortie
    a2 = sigmoide(z2)           # probabilité prédite
    return z1, a1, z2, a2


def calcul_perte(y, y_pred):
    """Entropie croisée binaire : forte quand le réseau se trompe avec assurance."""
    eps = 1e-8                  # évite log(0)
    return -np.mean(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))


# --------------------------------------------------------------
# Entraînement
# --------------------------------------------------------------
historique_perte = []

for epoque in range(1, nb_epoques + 1):

    # 1. PASSAGE AVANT (forward pass)
    z1, a1, z2, a2 = passage_avant(X)

    # 2. CALCUL DE LA PERTE
    perte = calcul_perte(y, a2)
    historique_perte.append(perte)

    # 3. RÉTROPROPAGATION (backpropagation) : calcul des gradients
    m = X.shape[0]                          # nombre d'exemples
    dz2 = a2 - y                            # erreur à la sortie (sigmoïde + entropie croisée)
    dW2 = a1.T @ dz2 / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m
    dz1 = (dz2 @ W2.T) * tanh_derivee(a1)   # erreur renvoyée vers la couche cachée
    dW1 = X.T @ dz1 / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    # Affichage détaillé de la première époque
    if epoque == 1:
        print("\n" + "=" * 58)
        print("DÉTAIL DE LA PREMIÈRE ÉPOQUE")
        print("=" * 58)
        print("1) Passage avant")
        print("   Activations de la couche cachée (a1) :\n", a1)
        print("   Sorties prédites (a2) :", a2.ravel())
        print("   Valeurs attendues     :", y.ravel())
        print(f"\n2) Perte : {perte:.4f}")
        print("\n3) Rétropropagation : poids W2 avant mise à jour :", W2.ravel())
        print("   Gradient dW2 :", dW2.ravel())

    # 4. MISE À JOUR DES POIDS (descente de gradient)
    W2 -= taux_apprentissage * dW2
    b2 -= taux_apprentissage * db2
    W1 -= taux_apprentissage * dW1
    b1 -= taux_apprentissage * db1

    if epoque == 1:
        print("   Poids W2 après mise à jour  :", W2.ravel())
        print("   (nouveau poids = ancien poids - taux x gradient)")
        print("\n" + "=" * 58)
        print("ÉVOLUTION DE LA PERTE")
        print("=" * 58)

    if epoque == 1 or epoque % 300 == 0:
        print(f"Époque {epoque:5d}  |  perte = {perte:.4f}")


# --------------------------------------------------------------
# Courbe de la perte
# --------------------------------------------------------------
plt.figure(figsize=(7, 4))
plt.plot(historique_perte)
plt.xlabel("Époque")
plt.ylabel("Perte (entropie croisée)")
plt.title("Réduction de la perte pendant l'entraînement - XOR")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("courbe_perte_xor.png", dpi=120)
plt.show()


# --------------------------------------------------------------
# Prédictions finales
# --------------------------------------------------------------
print("\n" + "=" * 58)
print("PRÉDICTIONS FINALES")
print("=" * 58)
_, _, _, probabilites = passage_avant(X)
classes = (probabilites >= 0.5).astype(int)

print(f"{'Entrée':<10}{'Attendu':>9}{'Probabilité':>14}{'Prédit':>9}")
for entree, attendu, proba, predit in zip(X, y, probabilites, classes):
    print(f"{str(entree):<10}{attendu[0]:>9}{proba[0]:>14.4f}{predit[0]:>9}")

precision = np.mean(classes == y)
print(f"\nPrécision : {precision:.0%}")
print(f"Perte : {historique_perte[0]:.4f} au départ -> {historique_perte[-1]:.4f} à la fin")

# Exemple de test isolé
exemple = np.array([[1, 0]])
_, _, _, p = passage_avant(exemple)
print(f"\nExemple de test {exemple[0]} : probabilité {p[0, 0]:.4f} "
      f"-> classe prédite {int(p[0, 0] >= 0.5)} (attendu : 1)")
