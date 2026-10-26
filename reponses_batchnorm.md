\# Batch Normalization : CNN de référence vs BN-CNN (MNIST)



Hyperparamètres identiques : SGD, taux 0,01, lots de 128, 20 époques. Code : `batchnorm\_mnist.py`.



| | Référence | BN-CNN |

|---|---|---|

| Perte d'entraînement après 1 époque | 1,3778 | 0,2826 |

| Époque où la validation dépasse 98 % | 14 | 2 |

| Perte d'entraînement finale | 0,0571 | 0,0190 |

| Précision de validation finale | 98,37 % | 98,92 % |

| Précision sur le test | 98,48 % | 98,90 % |

| Variation moyenne de la perte de validation | 0,0155 | 0,0169 |

| Durée par époque (CPU) | 10,3 s | 27,2 s |



!\[Courbes](batchnorm\_courbes.png)



\## Analyse



L'ajout de la Batch Normalization a nettement accéléré la convergence : avec des hyperparamètres identiques, la perte d'entraînement après la première époque passe de 1,378 à 0,283, et le BN-CNN dépasse 98 % de précision en validation dès la 2e époque, contre la 14e pour le CNN de référence. En normalisant les sorties de chaque couche, elle évite que les valeurs dérivent pendant l'entraînement, ce qui permet au réseau d'apprendre plus vite avec le même taux d'apprentissage. Le modèle final est aussi légèrement meilleur (98,90 % contre 98,48 % sur le test). En revanche, je n'ai pas observé de gain de stabilité : la perte de validation varie autant d'une époque à l'autre avec ou sans BN, sans doute parce que la BatchNorm utilise les statistiques de chaque lot pendant l'entraînement et des moyennes glissantes en validation. Enfin, chaque époque est environ 2,6 fois plus lente avec BN sur CPU, mais comme le BN-CNN atteint 98 % en 2 époques au lieu de 14, il y parvient environ 2,7 fois plus vite au total.

