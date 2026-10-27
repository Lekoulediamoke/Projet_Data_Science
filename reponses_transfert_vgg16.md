\# Transfert d'apprentissage avec VGG16 (CIFAR-10)



VGG16 pré-entraîné sur ImageNet, `include\_top=False`, images agrandies à 64 x 64.

Tête : Flatten -> Dense(256) -> Dropout(0,5) -> Dense(10, softmax).

5 000 images d'entraînement, 1 000 de validation, 2 000 de test. Code : `transfert\_vgg16.py`.



| | Phase 1 : base gelée | Phase 2 : affinage du bloc 5 |

|---|---|---|

| Taux d'apprentissage | 0,001 | 0,00001 |

| Paramètres entraînables | 527 114 | 7 606 538 |

| Précision entraînement | 73,94 % | 87,46 % |

| Précision validation | 69,30 % | 74,60 % |

| Précision test | 69,75 % | 74,10 % |

| Score F1 test (macro) | 0,693 | 0,739 |

| Durée (CPU) | 11,2 min | 13,4 min |



!\[Courbes](transfert\_vgg16\_courbes.png)



\## Analyse



Le transfert d'apprentissage avec VGG16 s'est révélé très efficace : en n'entraînant que la nouvelle tête de classification (527 114 paramètres, la base de 14,7 millions de paramètres restant gelée), le modèle atteint 69,75 % de précision sur le test dès la phase 1, contre 62,28 % pour mon meilleur CNN entraîné de zéro, qui disposait pourtant de deux fois plus d'images. Les caractéristiques apprises par VGG16 sur ImageNet (bords, textures, formes) sont donc réutilisables pour une nouvelle tâche. La phase 2, qui débloque le bloc 5 avec un taux d'apprentissage 100 fois plus faible, apporte un gain supplémentaire de 4,35 points (74,10 % sur le test, score F1 de 0,739), en adaptant les détecteurs de formes les plus spécifiques aux images de CIFAR-10. En revanche, elle fait apparaître les premiers signes de surapprentissage : l'écart entre entraînement et validation atteint 13 points (87,46 % contre 74,60 %) et la perte de validation remonte légèrement après la 2e époque, ce qui justifierait un arrêt anticipé. Enfin, les véhicules sont les mieux reconnus (score F1 jusqu'à 0,836 pour les bateaux), alors que les chats et les chiens, très semblables à cette résolution, restent les classes les plus difficiles.

