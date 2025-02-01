def nombres_premiers(n):
    if n < 2:
        return []
    # Initialiser une liste de booléens pour marquer les nombres premiers
    est_premier = [True] * (n + 1)
    est_premier[0] = est_premier[1] = False

    for i in range(2, int(n**0.5) + 1):
        if est_premier[i]:
            # Marquer les multiples de i comme non premiers
            for j in range(i*i, n+1, i):
                est_premier[j] = False

    # Retourner la liste des nombres premiers
    return [i for i, premier in enumerate(est_premier) if premier]

# Exemple
print(nombres_premiers(20))  # [2, 3, 5, 7, 11, 13, 17, 19]