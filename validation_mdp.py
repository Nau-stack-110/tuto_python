import re

def valider_mot_de_passe(mot_de_passe):
    if len(mot_de_passe) < 8:
        return False
    if not re.search("[A-Z]", mot_de_passe):
        return False
    if not re.search("[a-z]", mot_de_passe):
        return False
    if not re.search("[0-9]", mot_de_passe):
        return False
    if not re.search("[!@#$%^&*()]", mot_de_passe):
        return False
    return True

# Exemples
x = input("entrer votre mdp : ")
print(valider_mot_de_passe(x)) 