\# Affinage de BERT pour l'analyse de sentiments (SST-2)



Modèle : bert-base-uncased (Hugging Face), nouvelle tête à 2 classes.

Données : 3 000 phrases d'entraînement de SST-2, 872 phrases de validation, 64 jetons maximum.

Paramètres : Trainer, lots de 16, taux d'apprentissage 2e-5, 3 époques. Code : `bert\_sentiments.py`.



\## Suivi par époque



| Époque | Perte entraînement | Précision entraînement | Perte validation | Précision validation | F1 validation |

|---|---|---|---|---|---|

| 0 (avant affinage) | 0,6887 | 54,5 % | 0,6914 | 50,9 % | 0,638 |

| 1 | 0,1636 | 95,0 % | 0,2758 | 89,0 % | 0,888 |

| 2 | 0,0781 | 97,7 % | 0,3521 | 89,8 % | 0,900 |

| 3 | 0,0427 | 98,9 % | 0,4377 | 89,0 % | 0,892 |



!\[Courbes](bert\_courbes.png)



\## Évaluation finale (validation)



Précision : 88,99 % - Score F1 : 0,892



| | Prédit négatif | Prédit positif |

|---|---|---|

| Réel négatif | 379 | 49 |

| Réel positif | 47 | 397 |



!\[Matrice de confusion](bert\_matrice\_confusion.png)



\## Test sur mes propres phrases



| Phrase | Prédiction |

|---|---|

| I absolutely loved this film, the acting was brilliant. | positif (99,8 %) |

| What a waste of time, the plot made no sense. | négatif (99,6 %) |

| The movie was not bad at all. | positif (98,2 %) |

| I expected a masterpiece, but it was just boring. | négatif (99,6 %) |

| The soundtrack is great but the story is weak. | négatif (99,2 %) |



\## Analyse



Avant l'affinage, BERT obtient 50,9 % de précision en validation, soit le niveau du hasard : il connaît la langue mais pas encore la tâche. Une seule époque sur 3 000 phrases de SST-2 suffit à atteindre 89 %, et le modèle final obtient 88,99 % de précision et un score F1 de 0,892 sur les 872 phrases de validation, avec des erreurs équilibrées (49 faux positifs et 47 faux négatifs). Le suivi par époque montre cependant un surapprentissage dès la 2e époque : la perte d'entraînement continue de chuter (de 0,164 à 0,043) et la précision d'entraînement atteint 98,9 %, alors que la perte de validation remonte de 0,276 à 0,438 et que la précision de validation stagne autour de 89-90 %. Le modèle ne fait pas davantage d'erreurs, mais il devient trop sûr de lui, y compris quand il se trompe : 1 ou 2 époques auraient suffi, et un arrêt anticipé permettrait de garder automatiquement le meilleur modèle. Les tests sur mes propres phrases confirment ces deux aspects : BERT comprend la négation (« The movie was not bad at all » est jugée positive à 98,2 % malgré le mot « bad »), mais il classe l'avis mitigé « The soundtrack is great but the story is weak » comme négatif avec une confiance excessive de 99,2 %, faute de catégorie intermédiaire.

