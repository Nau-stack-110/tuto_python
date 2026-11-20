def fusionner_dictionnaires(dict1, dict2):
    return {**dict1, **dict2}

# Exemple
dict1 = {"a": 1, "b": 2}
dict2 = {"b": 3, "c": 4}
print(fusionner_dictionnaires(dict1, dict2))  # {"a": 1, "b": 3, "c": 4}