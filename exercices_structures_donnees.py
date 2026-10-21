# ==============================================================
# Exercices Python : structures de données et algorithmes
# ==============================================================


# --------------------------------------------------------------
# Question 1 : multiplier tous les éléments d'une liste
# --------------------------------------------------------------
print("=== Question 1 ===")

def multiplier_liste(liste):
    produit = 1                 # 1 est l'élément neutre de la multiplication
    for element in liste:
        produit *= element
    return produit

liste = [2, 3, 6]
print("Liste :", liste)
print("Résultat :", multiplier_liste(liste))


# --------------------------------------------------------------
# Question 2 : trier une liste de tuples selon leur dernier élément
# --------------------------------------------------------------
print("\n=== Question 2 ===")

tuples = [(2, 5), (1, 2), (4, 4), (2, 3), (2, 1)]
print("Liste :", tuples)

# key indique sur quoi trier : ici le dernier élément de chaque tuple
tuples.sort(key=lambda t: t[-1])
print("Résultat :", tuples)


# --------------------------------------------------------------
# Question 3 : combiner deux dictionnaires en additionnant
# les valeurs des clés communes
# --------------------------------------------------------------
print("\n=== Question 3 ===")

d1 = {'a': 100, 'b': 200, 'c': 300}
d2 = {'a': 300, 'b': 200, 'd': 400}

resultat = d2.copy()            # on part d'une copie pour ne pas modifier d2
for cle, valeur in d1.items():
    if cle in resultat:
        resultat[cle] += valeur  # clé commune : on additionne
    else:
        resultat[cle] = valeur   # clé absente : on l'ajoute

print("d1 :", d1)
print("d2 :", d2)
print("Résultat :", resultat)


# --------------------------------------------------------------
# Question 4 : dictionnaire {i: i*i} pour i de 1 à n
# --------------------------------------------------------------
print("\n=== Question 4 ===")

n = int(input("Entrez un nombre entier n : "))

carres = {}
for i in range(1, n + 1):       # n + 1 pour inclure n
    carres[i] = i * i

print(carres)


# --------------------------------------------------------------
# Question 5 : trier des tuples selon leur élément flottant
# (du plus grand au plus petit)
# --------------------------------------------------------------
print("\n=== Question 5 ===")

articles = [('item1', '12.20'), ('item2', '15.10'), ('item3', '24.5')]
print("Liste :", articles)

# Les prix sont des chaînes : on les convertit avec float() pour trier
articles_tries = sorted(articles, key=lambda t: float(t[1]), reverse=True)
print("Résultat :", articles_tries)


# --------------------------------------------------------------
# Question 6 : les ensembles (set)
# --------------------------------------------------------------
print("\n=== Question 6 ===")

# a) Créer un ensemble
ensemble = {0, 1, 2, 3, 4}
print("Ensemble créé :", ensemble)

# b) Itérer sur un ensemble
print("Parcours de l'ensemble :")
for element in ensemble:
    print(element)

# c) Ajouter des membres
ensemble.add(5)                 # ajoute un seul élément
ensemble.update([6, 7])         # ajoute plusieurs éléments
print("Après ajout de 5, 6 et 7 :", ensemble)

# d) Retirer des éléments
ensemble.remove(0)              # erreur si l'élément n'existe pas
ensemble.discard(10)            # aucune erreur si l'élément n'existe pas
print("Après retrait de 0 :", ensemble)
