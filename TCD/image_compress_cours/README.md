# Compression Lab — images, textes et audio

Application Streamlit pédagogique pour expérimenter la compression **sans perte** :

- **Images** : chaînes Delta / RLE / BWT / LZ77 / LZ78 / LZW, suivies ou non de
  Huffman ou d'un codage arithmétique, avec métriques de reconstruction.
- **Images par tuiles** : moteur Rust sans dépendance externe, qui choisit une
  chaîne par tuile et vérifie le round-trip.
- **Textes** : comparatif des pipelines du moteur Rust et vérification sans perte.
- **Audio** : FLAC sans perte pour le PCM entier WAV/AIFF, archivage exact
  `.rlab` pour les autres formats, avec vérification par hash.

Le PDF n'est pas pris en charge.

## Fichiers principaux

| Fichier | Rôle |
|---|---|
| `image_compression_app.py` | Application Streamlit et onglet Images |
| `compress_lab.py` | Algorithmes pédagogiques en Python |
| `text_lab.py` | Onglet Textes et appels au binaire Rust |
| `rust_image_lab.py` | Mode Images par tuiles |
| `audio_lab.py` | Onglet Audio (FLAC, `.rlab`, aperçu, forme d'onde) |
| `rust_codec/` | Moteur Rust sans perte (`compress`, `decompress`, `verify`, `compare`, `tile`) |
| `audios/original/` | Fichiers audio d'entrée |
| `audios/compressed/` | FLAC et archives `.rlab` produits |
| `audios/previews/` | Aperçus MP3 30 s et formes d'onde en cache |

Les dossiers `audios/` sont créés automatiquement au premier lancement.

## Prérequis

- Python 3 et les paquets de `requirements.txt` (Streamlit, NumPy, pandas,
  matplotlib, scikit-image, etc.).
- **Rust** (`cargo`) pour le moteur `.rlab` et le mode Images par tuiles.
  Le binaire est compilé automatiquement au premier usage s'il est absent.
- Outils système pour l'audio :
  - `ffmpeg` et `ffprobe` pour la détection du codec, l'aperçu, la forme d'onde
    et le décodage PCM ;
  - `flac` pour l'encodage et le test FLAC.

Sous Debian/Ubuntu :

```bash
sudo apt install ffmpeg flac
```

## Lancer l'application

```bash
cd TCD/image_compress_cours
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/streamlit run image_compression_app.py
```

Compiler le moteur Rust manuellement (facultatif) :

```bash
cd rust_codec
cargo build --release --offline
```

## Onglet Audio — règles de traitement

Le format réel est détecté avec `ffprobe`, **pas seulement avec l'extension**.
Un fichier `.ogg` contenant du FLAC est donc traité comme du FLAC, et un `.wav`
contenant du MP3 comme un format avec perte.

| Source détectée | Traitement proposé | Vérification |
|---|---|---|
| WAV/AIFF PCM entier 8/16/24/32 bits, 1 à 8 canaux | FLAC niveaux `-0`, `-5` (défaut) ou `-8` | `flac -t` + SHA-256 du PCM décodé en `s32le` |
| FLAC (dans `.flac`, `.ogg`, etc.) | Archivage exact `.rlab` | Vérification interne Rust + `verify` + comparaison octet par octet |
| WAV/AIFF PCM flottant ou PCM non supporté | Archivage exact `.rlab` | Idem |
| MP3, OGG Vorbis, Opus, AAC, PCM compandé… | Archivage exact `.rlab` uniquement | Idem |

Points importants :

- **FLAC garantit le PCM, pas l'identité octet pour octet du conteneur.**
  Les échantillons décodés sont strictement identiques, mais les chunks annexes
  du WAV/AIFF peuvent différer. Pour conserver le fichier d'origine tel quel,
  utiliser l'archivage `.rlab`.
- **MP3/OGG sont des formats avec perte.** Le fichier contient déjà une version
  dégradée du signal. Aucun algorithme ne peut restaurer le master ni améliorer
  la qualité. On peut seulement conserver les octets exacts dans un `.rlab`.
  Les métriques MAE/MSE/SNR/PSNR/SSIM par rapport au master ne sont pas
  mesurables sans ce master ; l'application affiche donc l'intégrité bit à bit.
- **Aucune limite arbitraire de taille ou de durée** n'est imposée. L'encodage
  FLAC, les hash PCM et la forme d'onde travaillent en flux. Un WAV de 163 Mo
  est traité sans charger tout le PCM en mémoire.
- **Aucun résultat plus gros que l'original n'est conservé.** Le FLAC ou le
  `.rlab` est supprimé dès qu'il n'apporte pas de gain (taille supérieure ou
  égale) et l'interface explique pourquoi. C'est fréquent pour un fichier déjà
  compressé ou très court.
- L'aperçu audio est un MP3 de 30 s à 128 kb/s, uniquement pour l'écoute.
  La forme d'onde est décodée en mono 8 kHz et mise en cache ; ce ne sont pas
  des résultats de compression.
- Le comparatif `-0 / -5 / -8` encode les trois niveaux, vérifie le plus petit
  et supprime les autres. Il est manuel pour ne pas ralentir les tests rapides.
- Le moteur Rust charge le fichier en mémoire pour l'archivage `.rlab` : sur un
  très gros WAV, préférer FLAC.

### Limite d'import du navigateur

Streamlit limite par défaut les imports à 200 Mo. Un WAV de 163 Mo passe, mais
pour un fichier plus gros, copiez-le directement dans `audios/original/` :

```bash
cp ~/Musique/mon_audio.wav audios/original/
```

La limite peut aussi être relevée dans `.streamlit/config.toml` :

```toml
[server]
maxUploadSize = 1024
```

## Métriques audio

Pour une conversion WAV/AIFF → FLAC vérifiée, l'application affiche :

- **MAE** et **MSE** : erreur moyenne et erreur quadratique moyenne entre les
  échantillons originaux et reconstruits ;
- **SNR** et **PSNR** : rapports signal/bruit, infinis lorsque l'erreur est nulle ;
- **SSIM temps-fréquence** : similarité définie sur le spectrogramme, car la
  SSIM classique est prévue pour des images 2D ; elle vaut 1 lorsque le PCM est
  identique ;
- **Erreur max**, **échantillons exacts** et **SHA-256 PCM**.

La preuve réelle du sans-perte reste le hash PCM identique, calculé en flux.
Pour MP3/OGG, seule l'intégrité bit à bit de l'archive est mesurable.

## Tests rapides

```bash
# Vérification syntaxique
venv/bin/python -m py_compile audio_lab.py image_compression_app.py text_lab.py rust_image_lab.py

# Moteur Rust
cd rust_codec && cargo build --release --offline

# Test fonctionnel audio (exemple)
venv/bin/python - <<'PY'
from pathlib import Path
import audio_lab as a

src = Path("audios/original/exemple.wav")
meta = a.probe_audio(src)
print(a.classify_audio(meta))
result = a._encode_and_verify_flac(src, Path("/tmp/exemple.flac"), 5)
print(result["verified"])
PY
```

## Documentation du moteur Rust

Voir `rust_codec/README.md` pour les commandes `compress`, `decompress`,
`verify`, `compare`, `tile`, `tile-decompress` et `tile-verify`.
