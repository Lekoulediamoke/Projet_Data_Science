\# Identifier et corriger le surapprentissage (CIFAR-10)



Sous-ensemble de 10 000 images d'entraînement, 5 000 de validation, 10 000 de test. Code : `surapprentissage\_cifar10.py`.



| | A : sans régularisation | B : augmentation + BN + Dropout | C : couches gelées |

|---|---|---|---|

| Précision entraînement | 94,14 % | 45,80 % | 61,45 % |

| Précision validation | 58,56 % | 50,24 % | 62,46 % |

| Perte de validation (min -> fin) | 1,111 -> 1,919 | 1,239 -> 1,342 | 1,097 -> 1,116 |

| Précision test | 59,06 % | 49,56 % | 62,28 % |

| Score F1 test (macro) | 0,588 | 0,465 | 0,605 |

| Durée | 3,5 min | 6,0 min | 1,0 min |



!\[Courbes](surapprentissage\_courbes.png)



\## Analyse



Le modèle de référence présente un surapprentissage net : 94,14 % de précision en entraînement contre 58,56 % en validation, et une perte de validation qui atteint son minimum à l'époque 7 (1,111) avant de remonter jusqu'à 1,919 pendant que la perte d'entraînement continue de baisser. La combinaison augmentation des données, Batch Normalization et Dropout (modèle B) supprime cet écart, mais elle régularise trop : avec seulement 20 époques et 10 000 images, le modèle tombe en sous-apprentissage (49,56 % sur le test, score F1 de 0,465) et ne reconnaît presque plus les chats, les chiens ni les oiseaux. La meilleure stratégie a été de geler les couches convolutionnelles du premier modèle, qui avaient déjà appris des motifs utiles, et de réentraîner uniquement les couches supérieures avec Dropout et augmentation (modèle C) : il obtient la meilleure précision sur le test (62,28 %) et le meilleur score F1 (0,605), avec des courbes d'entraînement et de validation qui restent proches, en une minute d'entraînement seulement. Cette expérience montre que la régularisation doit être dosée : trop peu entraîne un surapprentissage, trop en provoque un sous-apprentissage.

