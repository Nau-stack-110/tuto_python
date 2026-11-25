# Moteur Rust — compression sans perte

Codec pédagogique en Rust, **sans dépendance externe** (uniquement la bibliothèque standard).

## Compiler

```bash
cd rust_codec
cargo build --release --offline
```

Le binaire est produit dans `target/release/rust_codec`. L'application Streamlit le compile
automatiquement au premier usage s'il est absent.

## Étages disponibles

- `delta`
- `rle`
- `bwt`
- `lz77`
- `lz78`
- `lzw`
- `huffman`
- `arithmetic` (ou `arith`)

Huffman et l'arithmétique sont des codeurs terminaux : ils doivent être placés en dernier.
On peut combiner librement les préprocesseurs, par exemple `bwt+rle+huffman`,
`lz78+arithmetic` ou `lz77+huffman`.

`auto` essaie plusieurs chaînes et conserve le plus petit conteneur. Pour rester rapide,
il n'active BWT que sur les petits fichiers, et LZ78/LZW que sur les fichiers moyens.
Le comparatif exhaustif, lui, teste réellement toutes les combinaisons demandées.

## Utilisation

```bash
# Compresser (vérifie le round-trip pendant l'opération)
rust_codec compress entree.txt sortie.rlab lz77+huffman

# Décompresser
rust_codec decompress sortie.rlab entree_reconstruite.txt

# Vérifier les octets
rust_codec verify entree.txt sortie.rlab
```

## Comparatif exhaustif des combinaisons

`compare` énumère toutes les séquences ordonnées des préprocesseurs
**RLE, BWT, LZ77, LZ78, LZW** (sans répétition), suivies ou non de Huffman ou de
l'arithmétique. Chaque pipeline est encodé, décodé et vérifié. Le payload est mesuré
sans les métadonnées, comme dans `compress_lab.py`.

```bash
# 618 pipelines pour 4 étages max
rust_codec compare entree.txt 4
```

La sortie contient un résumé (`evaluated`, `gain_ge0`, `hidden`, `total_ms`) puis une
ligne `result=` par pipeline :
`pipeline`, `payload_bits`, `encode_ms`, `decode_ms`, `sans_perte`.
Le comparatif est limité à 2 Mo pour rester interactif.

L'interface Textes affiche le classement complet, le top par gain ou par rapidité, et
compare ces valeurs au modèle théorique de `compress_lab.py`.

## Images RGB par tuiles

Le mode `tile` compresse une image RGB brute (3 octets par pixel, sans en-tête) par tuiles.
En mode `auto`, chaque tuile teste plusieurs chaînes et garde la plus petite ; le conteneur
est ensuite décompressé et vérifié intégralement.

```bash
rust_codec tile image.rgb 1920 1080 128 image.rtile auto
rust_codec tile-decompress image.rtile image_reconstruite.rgb
rust_codec tile-verify image.rgb image.rtile
```

Le conteneur `.rtile` stocke les dimensions, la taille de tuile, l'empreinte de l'image
et, pour chaque tuile, la chaîne gagnante, ses métadonnées et son payload.

## Conteneurs

Le conteneur `.rlab` contient le pipeline, les tables/métadonnées et le payload final.
La taille du payload seul est affichée par `compress` (`payload_bytes`), comme dans
`compress_lab.py` ; `output_bytes` correspond au fichier complet.
