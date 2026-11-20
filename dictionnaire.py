# Créer un dictionnaire
d = {"nom": "Alice", "âge": 25, "ville": "Paris"}

# Propriétés
print("Nombre de clés :", len(d))            # Nombre de paires clé-valeur
print("Valeur associée à 'nom' :", d["nom"]) # Accéder à une valeur par clé
print("Clés du dictionnaire :", d.keys())    # Afficher les clés
print("Valeurs du dictionnaire :", d.values()) # Afficher les valeurs
print("Paires clé-valeur :", d.items())      # Afficher les paires clé-valeur

# Méthodes
age = d.get("âge", "Inconnu")                # Récupérer une valeur avec une valeur par défaut
print("Âge :", age)

d.update({"pays": "France"})                 # Ajouter une nouvelle paire clé-valeur
print("Après update :", d)

ville = d.pop("ville")                       # Supprimer et retourner la valeur associée à "ville"
print("Ville supprimée :", ville)
print("Après pop :", d)

cle, valeur = d.popitem()                    # Supprimer et retourner la dernière paire clé-valeur
print(f"Dernière paire supprimée : {cle} = {valeur}")
print("Après popitem :", d)

d.clear()                                    # Vider le dictionnaire
print("Après clear :", d)

# Copie d'un dictionnaire
d = {"a": 1, "b": 2}
copie = d.copy()
print("Copie du dictionnaire :", copie)

# Utilisation de setdefault
d.setdefault("c", 3)                         # Ajoute "c": 3 si "c" n'existe pas
print("Après setdefault :", d)

# Création d'un dictionnaire avec fromkeys
nouveau_dict = dict.fromkeys(["x", "y", "z"], 0)
print("Dictionnaire créé avec fromkeys :", nouveau_dict)