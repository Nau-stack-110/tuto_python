def compter_mots(chaine):
    # Diviser la chaîne en mots en utilisant les espaces comme séparateurs
    mots = chaine.split()
    return len(mots)

# Exemples
print(compter_mots("Bonjour le monde"))  # 3
print(compter_mots("Ceci est un test."))  # 4