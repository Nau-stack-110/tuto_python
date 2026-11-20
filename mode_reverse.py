chaine = input()
resultat = ""

# On parcourt chaque caractère avec son indice (i)
for i in range(len(chaine)):
    # Si l'indice est impair (1, 3, 5, etc.)
    if i % 2 != 0:
        resultat += chaine[i]

print(resultat)




s = input ()
r = s[1::2]
print(r)



Explication de chaine[1::2]
En Python, cette syntaxe utilise le slicing (découpage) qui fonctionne selon le format [début : fin : pas].

Pour chaine[1::2] :

1 (le début) : On commence à l'indice 1 (le deuxième caractère, car en informatique on commence à compter à 0).

Rien (la fin) : Comme il n'y a rien après le premier :, cela signifie qu'on va jusqu'à la fin de la chaîne.

2 (le pas) : On avance de 2 en 2 (on prend un caractère, on saute le suivant, etc.).

C'est une astuce très puissante, mais qui peut paraître un peu magique quand on débute.
