# Exercice : GPU et deep learning

## 1. Compréhension conceptuelle

Un CPU possède quelques cœurs très puissants, conçus pour exécuter des tâches variées les unes après les autres, alors qu'un GPU contient des milliers de petits cœurs capables d'effectuer la même opération sur beaucoup de données en même temps. Or l'entraînement d'un réseau de neurones repose sur des calculs simples répétés des millions de fois, ce qui convient parfaitement à ce calcul parallèle. Les opérations les mieux traitées par un GPU sont les multiplications de matrices (les sommes pondérées de chaque couche), les convolutions sur les images et le calcul des gradients pendant la rétropropagation, effectués sur des lots entiers d'exemples à la fois.

## 2. Vérification de l'environnement

Code exécuté : voir `verif_gpu.py`.

Sortie obtenue :

    CUDA disponible : False
    Périphérique GPU : Aucun GPU détecté

CUDA est la technologie de NVIDIA qui permet à PyTorch d'exécuter les calculs sur une carte graphique. La valeur `False` indique que PyTorch ne dispose d'aucun GPU utilisable : tous les calculs, y compris l'entraînement, seront effectués par le CPU. La commande `torch.version.cuda` renvoie `None`, ce qui confirme que la version de PyTorch installée est compilée pour le CPU uniquement. C'est suffisant pour de petits jeux de données comme Iris, mais l'entraînement sur des milliers d'images serait beaucoup plus lent ; on pourrait alors utiliser Google Colab, qui fournit un GPU gratuitement.

## 3. Question de réflexion

Avec 10 000 images, chaque époque demande des millions de multiplications de matrices et de convolutions sur les pixels ; un GPU les exécute en parallèle sur des lots entiers d'images, ce qui peut rendre l'entraînement plusieurs dizaines de fois plus rapide que sur un CPU. On peut ainsi faire plus d'époques et tester plus de réglages d'hyperparamètres dans le même temps.
