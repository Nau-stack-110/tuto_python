# Fonction pour créer une clé de substitution basée sur un texte de référence
def creer_cle_substitution(alphabet, texte_reference):
    # Supprimer les doublons dans le texte de référence tout en conservant l'ordre
    texte_unique = []
    for lettre in texte_reference:
        if lettre not in texte_unique and lettre in alphabet:
            texte_unique.append(lettre)
    
    # Ajouter les lettres restantes de l'alphabet
    for lettre in alphabet:
        if lettre not in texte_unique:
            texte_unique.append(lettre)
    
    # Créer un dictionnaire de substitution
    cle_substitution = {alphabet[i]: texte_unique[i] for i in range(len(alphabet))}
    return cle_substitution

# Fonction pour crypter un texte en utilisant la clé de substitution
def crypter_texte(texte, cle_substitution):
    texte_crypte = []
    for lettre in texte:
        if lettre in cle_substitution:
            texte_crypte.append(cle_substitution[lettre])
        else:
            texte_crypte.append(lettre)  # Garder les caractères non alphabétiques inchangés
    return ''.join(texte_crypte)

# Fonction principale
def main():
    # Définir l'alphabet normal
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    
    # Demander à l'utilisateur de saisir le texte de référence
    texte_reference = input("Entrez le texte de référence : ").lower()
    
    # Créer la clé de substitution
    cle_substitution = creer_cle_substitution(alphabet, texte_reference)
    
    # Afficher la clé de substitution
    print("\nClé de substitution :")
    for lettre_originale, lettre_cryptee in cle_substitution.items():
        print(f"{lettre_originale} -> {lettre_cryptee}")
    
    # Demander à l'utilisateur de saisir le texte à crypter
    texte_a_crypter = input("\nEntrez le texte à crypter : ").lower()
    
    # Crypter le texte
    texte_crypte = crypter_texte(texte_a_crypter, cle_substitution)
    
    # Afficher le résultat
    print("\nTexte crypté :")
    print(texte_crypte)

# Exécuter le programme
if __name__ == "__main__":
    main()