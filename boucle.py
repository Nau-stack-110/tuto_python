# En Python, les boucles sont utilisées pour répéter un bloc de code plusieurs fois.
# Les deux types de boucles les plus courants sont :

#     for : Pour itérer sur une séquence (liste, tuple, dictionnaire, etc.).

#     while : Pour répéter un bloc de code tant qu'une condition est vraie.

fruits = ["pomme", "banane", "cerise"]
for fruit in fruits:
    print(fruit)

    
for lettre in "Python":
    print(lettre)    




d = {"a": 1, "b": 2, "c": 3}

for cle in d:  # Itère sur les clés
    print(cle, d[cle])

for valeur in d.values():  # Itère sur les valeurs
    print(valeur)

for cle, valeur in d.items():  # Itère sur les paires clé-valeur
    print(cle, valeur)
    
    
        
for i in range(5):  # De 0 à 4
    print(i)
        


# Le bloc else est exécuté après que la boucle for a terminé son itération (sauf si la boucle est interrompue par break).
for i in range(3):
    print(i)
else:
    print("Boucle terminée.")    



compteur = 0
while compteur < 5:
    print(compteur)
    compteur += 1




compteur = 0
while compteur < 3:
    print(compteur)
    compteur += 1
else:
    print("Boucle terminée.")
    
    
    
    
while True:
    reponse = input("Entrez 'q' pour quitter : ")
    if reponse == 'q':
        break
# La boucle continue jusqu'à ce que l'utilisateur entre 'q'.







# Contrôle des boucles : break, continue, pass
#     break : Interrompt la boucle.

#     continue : Saute le reste du bloc de code et passe à l'itération suivante.

#     pass : Ne fait rien (utilisé comme placeholder).

for i in range(10):
    if i == 5:
        break
    print(i)
    


for i in range(5):
    if i == 2:
        continue
    print(i)



for i in range(3):
    if i == 1:
        pass  # Ne fait rien
    print(i)



    
for i in range(3):
    for j in range(2):
        print(f"i = {i}, j = {j}")
        
        

# Boucles avec compréhension de liste

    # Python permet de créer des listes, des dictionnaires,
    # des ensembles et des générateurs de manière concise en utilisant des boucles for dans une seule ligne.

carres = [x**2 for x in range(5)]
print(carres)


d = {x: x**2 for x in range(5)}
print(d)





