# Techniques de compression, une par fichier

Chaque fichier est exécutable seul et affiche les tableaux de la technique
correspondante, puis le résultat, la taille et le taux de réduction.

| Fichier | Technique | Tableaux affichés |
|---|---|---|
| `huffman.py` | Huffman | symboles, fréquences, probabilités, codes, longueurs, résultat binaire |
| `rle.py` | RLE | runs, bits par symbole et par longueur, représentation |
| `arithmetique.py` | Arithmétique | table 1 : `i, s_i, n_i, P(s_i), C(s_i-1), C(s_i)` avec `i` dès 1, symboles dans l'ordre d'apparition et valeurs tronquées à 4 décimales ; table 2 : `j, i, donnée, L_j, R_j` avec `j = 0`, `L0 = 0`, `R0 = 1`, valeurs tronquées à 4 décimales et payload binaire minimal |
| `bwt.py` | BWT | rotations possibles, rotations triées, dernière colonne, index primaire |
| `lz77.py` | LZ77 | buffer de recherche, buffer d'entrée, token, distance, longueur |
| `lz78.py` | LZ78 | construction du dictionnaire, tokens, dictionnaire final |
| `lzw.py` | LZW | encodage : `Indexe, Dictionnaire, Encodage` ; décodage : `Indexe, Dictionnaire, Décodage` ; résumé des bits |

## Commandes

```bash
python3 techniques/huffman.py
python3 techniques/huffman.py "ABRACADABRA"

python3 techniques/rle.py
python3 techniques/rle.py "AAAABBBCCDAA"

python3 techniques/arithmetique.py
python3 techniques/arithmetique.py "BANANA"

python3 techniques/bwt.py
python3 techniques/bwt.py "banana"

python3 techniques/lz77.py
python3 techniques/lz77.py --window 8 --lookahead 5

python3 techniques/lz78.py
python3 techniques/lz78.py "ABABABA"

python3 techniques/lzw.py
python3 techniques/lzw.py "ABABABA"
```

Options communes :

```bash
-f monfichier.txt      # lire un fichier
-d 4                   # chiffres après la virgule
--no-color             # désactiver les couleurs
```

Toutes les tailles affichées portent sur le payload. Chaque script vérifie
`encode → decode` et affiche `Vérification sans perte : OK`.
