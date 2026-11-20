def inverser_dictionnaire(d):
    # Créer un nouveau dictionnaire avec les clés et valeurs inversées
    return {v: k for k, v in d.items()}

# Exemple
print(inverser_dictionnaire({"a": 1, "b": 2, "c": 3}))  # {1: "a", 2: "b", 3: "c"}