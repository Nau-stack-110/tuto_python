nom = "Alice"
age = 21
print("Bonjour, je m'appelle %s et j'ai %d ans." % (nom, age))



nom = "Naunau"
age = 20
print("Bonjour, je m'appelle {} et j'ai {} ans.".format(nom, age))



print("Bonjour, je m'appelle {0} et j'ai {1} ans. {0} est mon prénom.".format(nom, age))



print("Bonjour, je m'appelle {nom} et j'ai {age} ans.".format(nom="Naunau", age=20))



nom = "Naunau"
age = 20
print(f"Bonjour, je m'appelle {nom} et j'ai {age} ans.")


a = 5
b = 10
print(f"La somme de {a} et {b} est {a + b}.")


pi = 3.14159
print(f"La valeur de pi arrondie est {pi:.2f}.")


pi = 3.14159
print("La valeur de pi arrondie est {:.2f}.".format(pi))


# Options de formatage courantes :

#     :.2f : Arrondir à 2 décimales.

#     :10d : Afficher un entier sur 10 caractères (alignement à droite).

#     :<10s : Afficher une chaîne sur 10 caractères (alignement à gauche).

#     :^10s : Centrer une chaîne sur 10 caractères.



donnees = {"nom": "Naunau", "age": 20}
print("Bonjour, je m'appelle {nom} et j'ai {age} ans.".format_map(donnees))


mots = ["Bonjour", "le", "monde"]
phrase = " ".join(mots)
print(phrase)


# str.split()
# Cette méthode permet de diviser une chaîne en une liste de sous-chaînes.
phrase = "Bonjour le monde"
mots = phrase.split()
print(mots)



texte = "Bonjour le monde"
nouveau_texte = texte.replace("monde", "Python")
print(nouveau_texte)


