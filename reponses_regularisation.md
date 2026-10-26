\# Point de contrôle 14 : augmentation des données et Dropout



Jeu de données : Sign Language MNIST (ASL). Deux CNN identiques entraînés 10 époques.

Code : `regularisation\_asl.py`.



| | Sans régularisation (A) | Augmentation + Dropout (B) |

|---|---|---|

| Précision d'entraînement | 100,00 % | 77,89 % |

| Précision sur le test | 90,44 % | 92,61 % |



\- Augmentation : rotation (environ +/- 15 degrés), zoom (+/- 10 %), décalage (+/- 10 %).

\- Dropout : 0,25 après chaque bloc de convolution, 0,5 avant la couche de sortie.



!\[Images augmentées](augmentation\_exemples.png)

!\[Courbes](regularisation\_courbes.png)



\## Analyse



J'ai choisi la rotation, le zoom et le décalage parce qu'ils reproduisent les variations réelles d'une photo de main (main un peu inclinée, plus ou moins proche de la caméra, pas parfaitement centrée) sans changer la lettre signée, contrairement au retournement horizontal, qui transformerait une main droite en main gauche. Combinés au Dropout, ils ont réduit le surapprentissage : la précision sur le jeu de test, composé de nouvelles photos, passe de 90,44 % à 92,61 %, alors que le modèle sans régularisation atteignait 100 % sur l'entraînement en mémorisant les images. En contrepartie, l'apprentissage est plus lent : la précision d'entraînement n'est que de 77,89 % après 10 époques, car le réseau voit des images plus difficiles et ne peut plus compter sur chaque neurone, ce qui suggère qu'avec davantage d'époques le modèle régularisé pourrait encore progresser.

