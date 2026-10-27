\# Classificateur de fleurs personnalisé (MobileNetV2)



Jeu de données : 3 670 photos de 5 espèces de fleurs (2 936 en entraînement, 734 en validation), redimensionnées en 160 x 160.

Modèle : MobileNetV2 pré-entraîné sur ImageNet, `include\_top=False`, extraction de caractéristiques (toutes les couches gelées).

Tête : GlobalAveragePooling2D -> Dense(128) -> Dropout(0,3) -> Softmax (5 classes). 15 époques. Code : `classificateur\_fleurs.py`.



| | Entraînement | Validation |

|---|---|---|

| Précision finale | 97,34 % | 87,33 % |

| Perte finale | 0,0789 | 0,4241 |

| Perte de validation minimale | | 0,3502 (époque 9) |

| Score F1 (macro) | | 0,869 |



| Classe | Précision | Rappel | Score F1 |

|---|---|---|---|

| Marguerite | 0,780 | 0,925 | 0,846 |

| Pissenlit | 0,934 | 0,895 | 0,914 |

| Rose | 0,795 | 0,849 | 0,821 |

| Tournesol | 0,904 | 0,911 | 0,908 |

| Tulipe | 0,913 | 0,808 | 0,857 |



!\[Courbes](fleurs\_courbes.png)

!\[Matrice de confusion](fleurs\_matrice\_confusion.png)



\## Analyse



Grâce à l'extraction de caractéristiques avec MobileNetV2 (toutes les couches gelées, seule la nouvelle tête de 164 613 paramètres étant entraînée), le classificateur atteint 87,33 % de précision en validation et un score F1 de 0,869 sur 5 espèces de fleurs, et dépassait déjà 84 % dès la première époque, ce qui montre que les caractéristiques apprises sur ImageNet se transfèrent bien à cette nouvelle tâche. Il n'y a pas de sous-apprentissage, mais un surapprentissage modéré apparaît à partir de l'époque 9 : la perte de validation y atteint son minimum (0,3502) puis oscille autour de 0,40-0,44 alors que celle d'entraînement continue de chuter, et la précision d'entraînement finit 10 points au-dessus de celle de validation (97,34 % contre 87,33 %). Ce surapprentissage reste léger, car la précision de validation se maintient autour de 87-88 % au lieu de baisser. Le pissenlit et le tournesol, aux formes et couleurs très caractéristiques, sont les mieux reconnus (scores F1 de 0,914 et 0,908), tandis que la rose et la tulipe, plus proches visuellement, sont les plus difficiles. Un arrêt anticipé vers l'époque 9, un Dropout un peu plus fort ou l'affinage des dernières couches de MobileNetV2 permettraient d'améliorer encore la généralisation.

