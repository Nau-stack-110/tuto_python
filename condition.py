# Exemple simple avec if-else
age = 18

if age >= 18:
    print("Vous êtes majeur.")
else:
    print("Vous êtes mineur.")
    
    

# Exemple avec if-elif-else
note = 75

if note >= 90:
    print("Grade : A")
elif note >= 80:
    print("Grade : B")
elif note >= 70:
    print("Grade : C")
elif note >= 60:
    print("Grade : D")
else:
    print("Grade : F")
    
    


# Exemple de conditions imbriquées
age = 20
nationalite = "Malagasy"

if age >= 18:
    if nationalite == "Malagasy":
        print("Vous pouvez voter à Madagascar.")
    else:
        print("Vous ne pouvez pas voter à Madagascar.")
else:
    print("Vous êtes trop jeune pour voter.")
    




# Exemple avec des opérateurs logiques
age = 25
salaire = 40000

if age >= 18 and salaire > 30000:
    print("Vous êtes éligible pour un prêt.")
else:
    print("Vous n'êtes pas éligible pour un prêt.")
    
    
    
# Exemple de condition ternaire
age = 20
statut = "Majeur" if age >= 18 else "Mineur"

print(statut)




# Exemple avec un dictionnaire
d = {"nom": "Naunau", "âge": 20}

if "nom" in d:
    print(f"Nom : {d['nom']}")
else:
    print("La clé 'nom' n'existe pas.")
    
    
    
    

# Exemple avec des chaînes de caractères
langage = "Python"

if langage == "Python":
    print("Vous utilisez Python.")
elif langage == "Java":
    print("Vous utilisez Java.")
else:
    print("Langage inconnu.")
    




# Exemple avec l'opérateur not
est_connecte = False

if not est_connecte:
    print("Veuillez vous connecter.")
else:
    print("Bienvenue !")
    
    
    


# Exemple avec une liste
fruits = ["pomme", "banane", "orange"]

if "banane" in fruits:
    print("La banane est dans la liste.")
else:
    print("La banane n'est pas dans la liste.")
    
    
    
    
# Exemple avec des conditions complexes
age = 22
est_etudiant = True

if (age >= 18 and age <= 25) and est_etudiant:
    print("Vous êtes éligible pour une réduction étudiante.")
else:
    print("Vous n'êtes pas éligible pour une réduction étudiante.")




# Variable avec la valeur None
resultat = None

if resultat is None:
    print("Aucun résultat trouvé.")
else:
    print("Résultat :", resultat)