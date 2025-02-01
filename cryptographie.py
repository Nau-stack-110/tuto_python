import string

# Définir une clé de substitution (par exemple, un alphabet décalé de 3 lettres)
def create_substitution_key(shift):
    alphabet = string.ascii_lowercase
    shifted_alphabet = alphabet[shift:] + alphabet[:shift]
    return dict(zip(alphabet, shifted_alphabet))

# Fonction pour crypter un texte en utilisant la clé de substitution
def encrypt(text, substitution_key):
    encrypted_text = []
    for char in text.lower():
        if char in substitution_key:
            encrypted_text.append(substitution_key[char])
        else:
            encrypted_text.append(char)  # Garder les caractères non alphabétiques inchangés
    return ''.join(encrypted_text)

# Exemple d'utilisation
if __name__ == "__main__":
    shift = 3  # Décalage de 3 lettres
    substitution_key = create_substitution_key(shift)
    
    text = input("entrer le text original : ")
    encrypted_text = encrypt(text, substitution_key)
    
    print(f"Texte original: {text}")
    print(f"Texte crypté: {encrypted_text}")
    
    
