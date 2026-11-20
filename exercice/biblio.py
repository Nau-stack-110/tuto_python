bibliotheque = []

def ajouter_livre(titre, auteur, annee):
    bibliotheque.append({"titre": titre, "auteur": auteur, "annee": annee})

def rechercher_par_titre(titre):
    return [livre for livre in bibliotheque if livre["titre"] == titre]

def livres_par_auteur(auteur):
    return [livre for livre in bibliotheque if livre["auteur"] == auteur]

def supprimer_livre(titre):
    global bibliotheque
    bibliotheque = [livre for livre in bibliotheque if livre["titre"] != titre]

# Exemple d'utilisation
ajouter_livre("1984", "George Orwell", 1949)
ajouter_livre("Le Petit Prince", "Antoine de Saint-Exupéry", 1943)
print(rechercher_par_titre("1984"))  # [{'titre': '1984', 'auteur': 'George Orwell', 'annee': 1949}]
print(livres_par_auteur("Antoine de Saint-Exupéry"))  # [{'titre': 'Le Petit Prince', 'auteur': 'Antoine de Saint-Exupéry', 'annee': 1943}]
supprimer_livre("1984")
print(bibliotheque)  # [{'titre': 'Le Petit Prince', 'auteur': 'Antoine de Saint-Exupéry', 'annee': 1943}]