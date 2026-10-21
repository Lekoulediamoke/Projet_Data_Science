# ==============================================================
# Calculatrice simple en Python
# ==============================================================


def calculatrice(num1, num2):
    """Demande un opérateur et renvoie le résultat de l'opération."""

    # 3. Demander l'opérateur à l'utilisateur
    operateur = input("Entrez l'opérateur (+, -, *, /) : ").strip()

    # 4. Choisir l'opération selon l'opérateur saisi
    if operateur == "+":
        resultat = num1 + num2
    elif operateur == "-":
        resultat = num1 - num2
    elif operateur == "*":
        resultat = num1 * num2
    elif operateur == "/":
        # 7. Cas de la division par zéro
        if num2 == 0:
            return "Erreur : division par zéro impossible."
        resultat = num1 / num2
    else:
        # 7. Cas d'un opérateur non valide
        return "Erreur : opérateur non valide. Utilisez +, -, * ou /."

    # Affiche 50 au lieu de 50.0 quand le résultat est un nombre entier
    if resultat == int(resultat):
        resultat = int(resultat)

    # 5. Retourner le résultat
    return resultat


def saisir_nombre(message):
    """Redemande tant que l'utilisateur n'entre pas un nombre valide."""
    while True:
        try:
            return float(input(message))
        except ValueError:
            print("Ce n'est pas un nombre valide. Réessayez.")


# Programme principal : la boucle while permet de faire plusieurs calculs
print("Bienvenue dans la calculatrice !")

continuer = "o"
while continuer == "o":
    nombre1 = saisir_nombre("Entrez le premier nombre : ")
    nombre2 = saisir_nombre("Entrez le deuxième nombre : ")

    # 6. Appeler la fonction et afficher le résultat
    resultat = calculatrice(nombre1, nombre2)

    if isinstance(resultat, str):      # un message d'erreur a été renvoyé
        print(resultat)
    else:
        print("Le résultat est :", resultat)

    continuer = input("\nVoulez-vous faire un autre calcul ? (o/n) : ").strip().lower()

print("Merci d'avoir utilisé la calculatrice. À bientôt !")
