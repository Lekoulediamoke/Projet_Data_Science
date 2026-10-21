import random

# 3. Générer un nombre entier aléatoire entre 1 et 100
nombre_secret = random.randint(1, 100)

print("Bienvenue dans le jeu Devinez le nombre !")
print("Je pense à un nombre entre 1 et 100. Pouvez-vous le deviner ?")

trouve = False

while not trouve:
    # 1. Demander la réponse du joueur (chaîne de caractères)
    reponse = input("Entrez votre réponse : ")

    # 2. Convertir la réponse en nombre entier
    estimation = int(reponse)

    # 4. Vérifier si l'estimation est trop élevée, trop basse ou correcte
    if estimation > nombre_secret:
        print("Votre réponse est trop élevée. Réessayez.")
    elif estimation < nombre_secret:
        print("Votre réponse est trop basse. Réessayez.")
    else:
        print("Félicitations ! Vous avez deviné le nombre correctement !")
        trouve = True
