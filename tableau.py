# Créer une liste
liste = [3, 1, 4, 1, 5, 9, 2, 6, 5]

# Propriétés
print("Longueur de la liste :", len(liste))  # Longueur de la liste
print("Premier élément :", liste[0])         # Accéder au premier élément
print("Sous-liste :", liste[2:5])            # Extraire une sous-liste

# Méthodes
liste.append(7)                              # Ajouter un élément à la fin
print("Après append :", liste)

liste.insert(2, 8)                           # Insérer un élément à l'index 2
print("Après insert :", liste)

liste.remove(1)                              # Supprimer la première occurrence de 1
print("Après remove :", liste)

element_pop = liste.pop(3)                   # Supprimer et retourner l'élément à l'index 3
print("Élément pop :", element_pop)
print("Après pop :", liste)

liste.sort()                                 # Trier la liste
print("Après sort :", liste)

liste.reverse()                              # Inverser la liste
print("Après reverse :", liste)

print("Nombre de 5 :", liste.count(5))       # Compter les occurrences de 5
print("Index de 9 :", liste.index(9))        # Trouver l'index de 9

copie = liste.copy()                         # Copier la liste
print("Copie de la liste :", copie)

liste.clear()                                # Vider la liste
print("Après clear :", liste)