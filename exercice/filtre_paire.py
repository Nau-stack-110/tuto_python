def filtrer_pairs(nombres):
    # Utiliser une compréhension de liste pour filtrer les nombres pairs
    return [n for n in nombres if n % 2 == 0]

# Exemple
print(filtrer_pairs([1, 2, 3, 4, 5, 6]))  # [2, 4, 6]