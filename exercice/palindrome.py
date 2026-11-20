def est_palindrome(chaine):
    # Nettoyer la chaîne : enlever les espaces et mettre en minuscule
    chaine_propre = ''.join(chaine.split()).lower()
    # Comparer la chaîne avec son inverse
    return chaine_propre == chaine_propre[::-1]

# Exemples
print(est_palindrome("Esope reste ici et se repose"))  # True
print(est_palindrome("Python"))  # False