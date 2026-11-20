# Déclaration de variables
nom = "Alice"          # Chaîne de caractères (str)
age = 25               # Entier (int)
taille = 1.75          # Nombre à virgule flottante (float)
est_etudiant = True    # Booléen (bool)



# Affectation multiple
nom, age, taille = "Bob", 30, 1.80


# Affectation de la même valeur à plusieurs variables
x = y = z = 10


# Variables avec des types complexes
liste = [1, 2, 3]                # Liste (list)
dictionnaire = {"a": 1, "b": 2}  # Dictionnaire (dict)
tuple_exemple = (4, 5, 6)        # Tuple (tuple)
ensemble = {1, 2, 3}             # Ensemble (set)



# Variables avec des types personnalisés
from typing import List, Dict, Tuple

nombres: List[int] = [1, 2, 3]                     # Liste d'entiers
coordonnees: Tuple[float, float] = (48.8566, 2.3522) # Tuple de flottants
informations: Dict[str, int] = {"age": 20, "taille": 180} # Dictionnaire



# Variable globale
x = 10
def ma_fonction():
    # Variable locale
    y = 5
    print("Variable locale y :", y)
    print("Variable globale x :", x)

ma_fonction()



# Réaffectation de variables
a = 10
print("a avant réaffectation :", a)

a = "Bonjour"
print("a après réaffectation :", a)



# Constante (convention)
PI = 3.14159
TAUX_TVA = 0.20
print("Valeur de PI :", PI)



# Déballage de tuple
coordonnees = (48.8566, 2.3522)
latitude, longitude = coordonnees

print("Latitude :", latitude)
print("Longitude :", longitude)




# Bonnes pratiques pour la déclaration de variables

#     Noms significatifs : Utilisez des noms de variables descriptifs.

#         Exemple : nombre_etudiants au lieu de n.

#     Style de nommage : Utilisez le snake_case pour les noms de variables.

#         Exemple : ma_variable au lieu de maVariable.

#     Évitez les mots-clés réservés : Ne pas utiliser des mots-clés Python comme if, else, for, etc.

#     Constantes : Utilisez des majuscules pour les constantes (convention).

#         Exemple : TAUX_TVA = 0.20.