#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Laboratoire de compression d'images utilisant les algorithmes de compress_lab.py.
Placer ce fichier dans le même dossier que compress_lab.py.

Deux modes :
  • Chaîne unique   : Delta / RLE / BWT / LZ77 / LZ78 / LZW, suivis (ou non) de Huffman
                      ou de l'Arithmétique (codeur terminal), avec mesures détaillées.
                      Option « tuiles » : compresse l'image ENTIÈRE, sans perte, bloc par bloc.
  • Comparatif      : évalue automatiquement toutes les chaînes (solo, combinées,
                      avec Huffman / Arithmétique), comme le fait la console.

Mesures : gain, ratio, temps d'encodage / décodage / total, latence, débit.
"""

from math import perm
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Recherche compress_lab.py dans le dossier du projet et ses dossiers parents.
for candidate in Path(__file__).resolve().parents:
    if (candidate / "compress_lab.py").is_file():
        sys.path.insert(0, str(candidate))
        break
import compress_lab as lab

# LZ77/LZ78 terminent leur flux par un marqueur unique (lab.NO_SYMBOL), reconnu par identité.
# Après sérialisation (pickle) dans le .clab, il devenait une copie non reconnue et la
# décompression échouait dès qu'une tuile se terminait par ce marqueur. On indique à pickle
# de toujours le restaurer comme l'objet unique du module.
if not hasattr(lab._NoSymbol, "__reduce__") or lab._NoSymbol.__reduce__ is object.__reduce__:
    lab._NoSymbol.__reduce__ = lambda self: "NO_SYMBOL"


ROOT = Path(__file__).resolve().parent
ORIGINAL_DIR = ROOT / "images" / "original"
COMPRESSED_DIR = ROOT / "images" / "compressed"
ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
COMPRESSED_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

STAGE_NAMES = ["Delta", "RLE", "BWT", "LZ77", "LZ78", "LZW"]
CODER_NAMES = ["Huffman", "Arithmétique"]

# Nombre maximal de symboles acceptés EN ENTRÉE de chaque algorithme.
# Les implémentations de compress_lab.py sont en Python pur (pédagogiques) :
# au-delà, le temps de calcul explose.
MAX_SYMBOLS = 1_000_000
LIMITS = {
    "Delta": None,
    "RLE": None,
    "BWT": 5_000,            # tri de toutes les rotations
    "LZ77": 100_000,
    "LZ78": 100_000,
    "LZW": 100_000,
    "Huffman": 1_000_000,
    "Arithmétique": 30_000,  # entiers de plusieurs millions de bits
}

RESIZE_CHOICES = ["Original", 32, 64, 128, 256, 512, 1024]
TILE_CHOICES = [16, 32, 64, 128]
DOWNLOAD_MAX_BYTES = 200_000_000

st.set_page_config(page_title="Image Compression Lab", page_icon="🗜️", layout="wide")
st.title("🗜️ Image Compression Lab")
st.caption("Laboratoire pédagogique — RLE, BWT, LZ77, LZ78, LZW, Huffman et codage arithmétique")


# ──────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ──────────────────────────────────────────────────────────────────────────────
def stretch(fn, *args, **kwargs):
    """Étire un widget sur toute la largeur, quelle que soit la version de Streamlit
    (width="stretch" pour les versions récentes, use_container_width sinon)."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


def gain_pct(raw_bits, bits):
    """Gain en % par rapport aux données brutes (négatif = expansion)."""
    return (raw_bits - bits) / raw_bits * 100 if raw_bits else 0.0


def fmt_ms(ms):
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    return f"{ms:.2f} ms"


def check_limit(name, size):
    limit = LIMITS.get(name)
    if limit is not None and size > limit:
        raise ValueError(
            f"{name} est limité à {limit:,} symboles en entrée (reçu : {size:,}). "
            "Active le mode « tuiles » avec des tuiles plus petites, réduis l'image "
            "testée, ou place avant lui un algorithme qui réduit le flux (ex. RLE)."
        )


@st.cache_resource(show_spinner="Chargement de l'image…")
def load_array(path_str, mtime, max_side):
    """Charge l'image en RGB, la réduit si demandé, et prépare un aperçu léger.

    cache_resource (et non cache_data) : évite de recopier un tableau de 200 Mo
    à chaque interaction. Les objets retournés ne doivent pas être modifiés.
    """
    with Image.open(path_str) as im:
        rgb = im.convert("RGB")
        full_size = rgb.size
        if max_side and max(rgb.size) > max_side:
            rgb.thumbnail((max_side, max_side), Image.LANCZOS)
        array = np.asarray(rgb, dtype=np.uint8)
        preview = rgb.copy()
        preview.thumbnail((1200, 1200), Image.LANCZOS)
        preview_array = np.asarray(preview, dtype=np.uint8)
    return array, full_size, preview_array


def delta_encode(stream):
    """Prédiction « pixel de gauche » : chaque valeur est remplacée par sa différence
    (modulo 256) avec la même composante du pixel précédent. Réversible, et les
    résidus se concentrent autour de 0 : Huffman/Arithmétique compressent bien mieux.
    Les autres algorithmes de compress_lab.py sont faits pour du texte ; c'est
    cette étape qui permet de bien compresser une photo."""
    arr = np.asarray(stream, dtype=np.uint8)
    out = arr.copy()
    if arr.size > 3:
        out[3:] = arr[3:] - arr[:-3]
    # même nombre de symboles de 8 bits : le gain vient de l'étape entropique suivante
    return out.tolist(), {}, int(arr.size) * 8


def delta_decode(stream, meta):
    arr = np.asarray(stream, dtype=np.uint8)
    n = int(arr.size)
    pad = (-n) % 3
    if pad:
        arr = np.concatenate([arr, np.zeros(pad, dtype=np.uint8)])
    restored = np.cumsum(arr.reshape(-1, 3), axis=0, dtype=np.uint8).reshape(-1)
    return restored[:n].tolist()


def get_algorithms():
    """Associe les noms de l'interface aux fonctions de compress_lab.py."""
    return {
        "Delta": {"encode": delta_encode, "decode": delta_decode},
        "RLE": {"encode": lab.rle_seq_encode, "decode": lab.rle_seq_decode},
        "BWT": {"encode": lab.bwt_seq_encode, "decode": lab.bwt_seq_decode},
        "LZ77": {
            "encode": lab.lz77_stage_encode,
            "decode": lambda stream, meta: lab.lz77_seq_decode(stream),
        },
        "LZ78": {
            "encode": lab.lz78_stage_encode,
            "decode": lambda stream, meta: lab.lz78_seq_decode(stream),
        },
        "LZW": {
            "encode": lab.lzw_stage_encode,
            "decode": lambda stream, meta: lab.lzw_seq_decode(stream, meta),
        },
    }


def get_coders():
    return {"Huffman": lab.Huffman, "Arithmétique": lab.Arith}


def _pack_entropy(name, meta):
    """Range les métadonnées du codeur dans une forme compacte pour le conteneur .clab.

    • Huffman : la chaîne de '0'/'1' (1 octet par bit) devient de vrais octets. Si les
      symboles sont des octets (0..255), la table (symbole, longueur) est stockée en
      deux petits tableaux d'octets au lieu d'une liste de tuples Python.
    """
    if name == "Huffman":
        bits = meta["bits"]
        n = len(bits)
        data = int(bits, 2).to_bytes((n + 7) // 8, "big") if n else b""
        table = meta["table"]
        compact = all(type(sym) is int and 0 <= sym <= 255 and 0 < length <= 255 for sym, length in table)
        packed = {"nbits": n, "data": data, "payload": meta["payload"]}
        if compact:
            packed["syms"] = bytes(sym for sym, _ in table)
            packed["lens"] = bytes(length for _, length in table)
        else:
            packed["table"] = table
        return packed
    return meta


def _unpack_entropy(name, packed):
    if name == "Huffman":
        n = packed["nbits"]
        bits = format(int.from_bytes(packed["data"], "big"), f"0{n}b") if n else ""
        if "syms" in packed:
            table = list(zip(packed["syms"], packed["lens"]))
        else:
            table = packed["table"]
        return {"table": table, "bits": bits, "payload": packed["payload"]}
    return packed


# ──────────────────────────────────────────────────────────────────────────────
# Mode « chaîne unique »
# ──────────────────────────────────────────────────────────────────────────────
def compress_pixels(pixel_array, selected_stages, entropy_name):
    """Encode les pixels, sérialise dans un conteneur .clab, puis décode depuis ce
    conteneur. Chaque phase est chronométrée.

    La vérification de chaque étape se fait pendant la vraie décompression
    (on compare ce qui est décodé au flux d'entrée mémorisé de l'étape) : pas de
    décodage supplémentaire, donc pas de temps perdu.
    """
    algorithms = get_algorithms()
    coders = get_coders()

    original = pixel_array.astype(np.uint8, copy=False).reshape(-1)
    raw_bits = int(original.size) * 8
    stream = original.tolist()

    inputs, encoded_stages, rows = [], [], []
    encode_s = 0.0
    payload_bits = raw_bits

    # --- Compression : préprocesseurs -----------------------------------------
    for name in selected_stages:
        check_limit(name, len(stream))
        started = time.perf_counter()
        new_stream, meta, est_bits = algorithms[name]["encode"](stream)
        elapsed = time.perf_counter() - started
        encode_s += elapsed

        inputs.append(stream)
        encoded_stages.append({"name": name, "meta": meta})
        payload_bits = int(est_bits)
        rows.append({"name": name, "before": len(stream), "after": len(new_stream),
                     "bits": payload_bits, "enc_s": elapsed, "dec_s": 0.0, "ok": None})
        stream = new_stream

    # --- Compression : codeur entropique terminal -----------------------------
    entropy_meta = None
    if entropy_name:
        check_limit(entropy_name, len(stream))
        started = time.perf_counter()
        entropy_meta = coders[entropy_name].enc(stream)
        elapsed = time.perf_counter() - started
        encode_s += elapsed
        payload_bits = int(entropy_meta["payload"])
        rows.append({"name": entropy_name, "before": len(stream), "after": len(stream),
                     "bits": payload_bits, "enc_s": elapsed, "dec_s": 0.0, "ok": None})
    final_stream = stream

    # --- Sérialisation du conteneur .clab -------------------------------------
    started = time.perf_counter()
    package = {
        "format": "ImageCompressionLab-1",
        "shape": tuple(int(x) for x in pixel_array.shape),
        "stages": encoded_stages,
        "entropy": entropy_name,
        "entropy_meta": _pack_entropy(entropy_name, entropy_meta) if entropy_meta else None,
        "final_stream": None if entropy_meta is not None else final_stream,
    }
    blob = pickle.dumps(package, protocol=5)
    serialize_s = time.perf_counter() - started

    # --- Décompression : repart du conteneur réellement sérialisé -------------
    started = time.perf_counter()
    loaded = pickle.loads(blob)
    deserialize_s = time.perf_counter() - started

    decode_s = 0.0
    decoded = loaded["final_stream"]
    if loaded["entropy"]:
        started = time.perf_counter()
        decoded = coders[loaded["entropy"]].dec(_unpack_entropy(loaded["entropy"], loaded["entropy_meta"]))
        elapsed = time.perf_counter() - started
        decode_s += elapsed
        rows[-1]["dec_s"] = elapsed
        rows[-1]["ok"] = decoded == final_stream
        if not rows[-1]["ok"]:
            raise ValueError(f"Échec de vérification du décodage {loaded['entropy']}.")

    for idx in reversed(range(len(loaded["stages"]))):
        stage = loaded["stages"][idx]
        started = time.perf_counter()
        decoded = algorithms[stage["name"]]["decode"](decoded, stage["meta"])
        elapsed = time.perf_counter() - started
        decode_s += elapsed
        rows[idx]["dec_s"] = elapsed
        rows[idx]["ok"] = decoded == inputs[idx]
        if not rows[idx]["ok"]:
            raise ValueError(f"Échec de vérification après l'étape {stage['name']}.")

    decoded_array = np.asarray(decoded, dtype=np.uint8)
    if decoded_array.size != original.size:
        raise ValueError(f"Taille reconstruite incorrecte : {decoded_array.size} au lieu de {original.size}.")
    decoded_array = decoded_array.reshape(loaded["shape"])
    exact = bool(np.array_equal(pixel_array, decoded_array))

    return {
        "decoded_array": decoded_array,
        "rows": rows,
        "raw_bits": raw_bits,
        "payload_bits": payload_bits,
        "exact": exact,
        "blob": blob,
        "blob_size": len(blob),
        "tiles": None,
        "compress_s": encode_s + serialize_s,
        "decompress_s": deserialize_s + decode_s,
        "encode_s": encode_s,
        "decode_s": decode_s,
        "serialize_s": serialize_s,
        "deserialize_s": deserialize_s,
    }


def compress_tiled(pixel_array, selected_stages, entropy_name, tile, clab_path, progress=None):
    """Compresse l'image ENTIÈRE, sans perte, bloc par bloc (tuiles tile × tile pixels).

    Chaque tuile est compressée puis décompressée indépendamment avec compress_pixels() :
    la mémoire reste bornée et les limites de taille des algorithmes s'appliquent à la
    tuile, pas à l'image. Les tuiles sont écrites une à une dans le fichier .clab.
    Contrepartie : les métadonnées (tables de Huffman, alphabets...) sont répétées dans
    chaque tuile, et les statistiques ne sont plus globales.
    """
    height, width, _ = pixel_array.shape
    n_tiles = -(-height // tile) * -(-width // tile)
    decoded = np.empty_like(pixel_array)

    sums = {k: 0.0 for k in ("compress_s", "decompress_s", "encode_s", "decode_s",
                             "serialize_s", "deserialize_s")}
    raw_bits = payload_bits = 0
    agg_rows = None
    exact_all, expanded, gain_min, gain_max, done = True, 0, None, None, 0
    started_all = time.perf_counter()
    last_update = 0.0

    with open(clab_path, "wb") as fh:
        pickle.dump({"format": "ImageCompressionLab-tiles-1", "shape": tuple(pixel_array.shape),
                     "tile": tile, "stages": list(selected_stages), "entropy": entropy_name,
                     "n_tiles": n_tiles}, fh, protocol=5)
        for ty in range(0, height, tile):
            for tx in range(0, width, tile):
                block = pixel_array[ty:ty + tile, tx:tx + tile]
                r = compress_pixels(block, selected_stages, entropy_name)
                decoded[ty:ty + tile, tx:tx + tile] = r["decoded_array"]
                pickle.dump(((ty, tx), r["blob"]), fh, protocol=5)

                for k in sums:
                    sums[k] += r[k]
                raw_bits += r["raw_bits"]
                payload_bits += r["payload_bits"]
                exact_all = exact_all and r["exact"]
                g = gain_pct(r["raw_bits"], r["payload_bits"])
                gain_min = g if gain_min is None else min(gain_min, g)
                gain_max = g if gain_max is None else max(gain_max, g)
                if r["payload_bits"] > r["raw_bits"]:
                    expanded += 1

                if agg_rows is None:
                    agg_rows = [dict(row, ok=True) for row in r["rows"]]
                else:
                    for acc, row in zip(agg_rows, r["rows"]):
                        for key in ("before", "after", "bits", "enc_s", "dec_s"):
                            acc[key] += row[key]
                done += 1
                now = time.perf_counter()
                if progress and (now - last_update > 0.25 or done == n_tiles):
                    last_update = now
                    elapsed = now - started_all
                    progress(done, n_tiles, elapsed, elapsed / done * (n_tiles - done))

    exact = bool(exact_all and np.array_equal(pixel_array, decoded))
    return {
        "decoded_array": decoded,
        "rows": agg_rows or [],
        "raw_bits": raw_bits,
        "payload_bits": payload_bits,
        "exact": exact,
        "blob": None,
        "blob_size": clab_path.stat().st_size,
        "tiles": {"size": tile, "count": n_tiles, "gain_min": gain_min,
                  "gain_max": gain_max, "expanded": expanded},
        **sums,
    }


def render_single(res):
    """Affiche les résultats d'une exécution en chaîne unique."""
    original_array = res["original_array"]
    decoded_array = res["decoded_array"]
    height, width, _ = original_array.shape
    n_pixels = height * width
    raw_bytes = original_array.nbytes
    payload_bits = res["payload_bits"]
    payload_bytes = (payload_bits + 7) // 8
    blob_size = res["blob_size"]

    st.subheader("4. Résultat visuel")
    col1, col2 = st.columns(2)
    with col1:
        stretch(st.image, res["preview_original"], caption="Image testée")
    with col2:
        stretch(st.image, res["preview_rebuilt"], caption="Image reconstruite")

    if res["exact"]:
        st.success("Vérification sans perte réussie : les matrices de pixels sont identiques.")
    else:
        st.error("La reconstruction diffère de l'image originale.")

    st.subheader("5. Matrice reconstruite")
    m = res["matrix_limit"]
    stretch(st.dataframe, decoded_array[:m, :m, :].tolist())

    st.subheader("6. Détails des étapes")
    if res["rows"]:
        table = []
        for r in res["rows"]:
            table.append({
                "Étape": r["name"],
                "Symboles avant": r["before"],
                "Symboles après": r["after"],
                "Payload (bits)": r["bits"],
                "Gain cumulé (%)": gain_pct(res["raw_bits"], r["bits"]),
                "Encodage (ms)": r["enc_s"] * 1000,
                "Décodage (ms)": r["dec_s"] * 1000,
                "Latence encodage (µs/symbole)": r["enc_s"] * 1e6 / max(r["before"], 1),
                "Latence décodage (µs/symbole)": r["dec_s"] * 1e6 / max(r["before"], 1),
                "Vérification": "OK" if r["ok"] else "ÉCHEC",
            })
        stretch(st.dataframe, pd.DataFrame(table), hide_index=True, column_config={
            "Gain cumulé (%)": st.column_config.NumberColumn(format="%.4f"),
            "Encodage (ms)": st.column_config.NumberColumn(format="%.3f"),
            "Décodage (ms)": st.column_config.NumberColumn(format="%.3f"),
            "Latence encodage (µs/symbole)": st.column_config.NumberColumn(format="%.3f"),
            "Latence décodage (µs/symbole)": st.column_config.NumberColumn(format="%.3f"),
        })
        st.caption("Gain cumulé = gain du payload à la sortie de l'étape, par rapport aux pixels RGB bruts. "
                   "Latence = temps moyen par symbole entrant dans l'étape.")
    else:
        st.info("Aucun algorithme sélectionné : les pixels sont stockés tels quels.")

    st.subheader("7. Mesures de compression")

    # ---- Tailles et gain
    saved_payload = raw_bytes - payload_bytes
    gain_payload = gain_pct(res["raw_bits"], payload_bits)
    ratio = (raw_bytes / payload_bytes) if payload_bytes else 0.0
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Pixels RGB bruts", f"{raw_bytes:,} octets")
    a2.metric("Payload compressé", f"{payload_bytes:,} octets")
    a3.metric("Gain (payload)", f"{gain_payload:.4f} %", delta=f"{saved_payload:+,} octets")
    a4.metric("Ratio (payload)", f"{ratio:.2f}:1")

    saved_container = raw_bytes - blob_size
    gain_container = gain_pct(raw_bytes, blob_size)
    b1, b2, b3 = st.columns(3)
    b1.metric("Conteneur .clab réel", f"{blob_size:,} octets")
    b2.metric("Gain (conteneur vs pixels bruts)", f"{gain_container:.4f} %", delta=f"{saved_container:+,} octets")
    if res["resized"]:
        b3.metric("Gain vs fichier original", "—", help="Non comparable : l'image a été réduite avant le test.")
    else:
        orig_size = res["original_file_size"]
        b3.metric("Gain (conteneur vs fichier original)", f"{gain_pct(orig_size, blob_size):.4f} %",
                  delta=f"{orig_size - blob_size:+,} octets")

    # ---- Temps et latence
    comp_ms, decomp_ms = res["compress_s"] * 1000, res["decompress_s"] * 1000
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temps de compression", fmt_ms(comp_ms),
              help=f"Encodage {fmt_ms(res['encode_s'] * 1000)} + sérialisation {fmt_ms(res['serialize_s'] * 1000)}")
    c2.metric("Temps de décompression", fmt_ms(decomp_ms),
              help=f"Décodage {fmt_ms(res['decode_s'] * 1000)} + désérialisation {fmt_ms(res['deserialize_s'] * 1000)}")
    c3.metric("Temps total d'exécution", fmt_ms(res["total_s"] * 1000),
              help="Compression + décompression + vérifications + sauvegarde des fichiers.")
    c4.metric("Débit de compression",
              f"{raw_bytes / 1024 / res['compress_s']:,.1f} Ko/s" if res["compress_s"] else "—")

    d1, d2, d3 = st.columns(3)
    d1.metric("Latence de compression", f"{res['compress_s'] * 1e6 / n_pixels:,.2f} µs/pixel")
    d2.metric("Latence de décompression", f"{res['decompress_s'] * 1e6 / n_pixels:,.2f} µs/pixel")
    d3.metric("Débit de décompression",
              f"{raw_bytes / 1024 / res['decompress_s']:,.1f} Ko/s" if res["decompress_s"] else "—")

    if res["tiles"]:
        t = res["tiles"]
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Tuiles", f"{t['count']:,}", help=f"Tuiles de {t['size']} × {t['size']} pixels")
        e2.metric("Gain min. d'une tuile", f"{t['gain_min']:.2f} %")
        e3.metric("Gain max. d'une tuile", f"{t['gain_max']:.2f} %")
        e4.metric("Tuiles en expansion", f"{t['expanded']:,}", help="Tuiles dont le payload dépasse les pixels bruts.")

    if gain_payload < 0:
        st.warning("Le payload est plus volumineux que les pixels bruts : cette combinaison "
                   "n'est pas adaptée à cette image (gain négatif).")
    if gain_container < 0:
        st.warning("Le conteneur .clab est plus volumineux que les pixels bruts.")
    elif gain_payload - gain_container > 15:
        st.warning(
            f"Le conteneur réel gagne beaucoup moins que le payload ({gain_container:.1f} % contre "
            f"{gain_payload:.1f} %) : les tables et métadonnées pèsent lourd. C'est fréquent avec des "
            "tuiles trop petites, ou quand Huffman code des jetons LZ77/LZ78 (milliers de symboles "
            "distincts). Essaie des tuiles plus grandes ou la chaîne Delta → Huffman."
        )
    st.caption(
        "Le gain (payload) suit la méthode de la console : taille des seules données compressées, "
        "sans les métadonnées. Le conteneur .clab inclut en plus les tables et métadonnées nécessaires "
        "à la reconstruction ; il est sérialisé avec pickle pour ce laboratoire pédagogique, ce n'est pas "
        "un format universel. Sans codeur Huffman/Arithmétique, le flux intermédiaire est stocké sous forme "
        "d'objets Python, d'où un conteneur nettement plus gros que le payload estimé. "
        "N'ouvre jamais un fichier pickle provenant d'une source inconnue."
    )
    st.write(f"Flux compressé enregistré : `{res['clab_path'].relative_to(ROOT)}`")
    if res["blob"] is not None:
        st.download_button("Télécharger le flux compressé (.clab)", data=res["blob"],
                           file_name=res["clab_path"].name, mime="application/octet-stream")
    elif blob_size <= DOWNLOAD_MAX_BYTES:
        st.download_button("Télécharger le flux compressé (.clab)", data=res["clab_path"].read_bytes(),
                           file_name=res["clab_path"].name, mime="application/octet-stream")
    else:
        st.caption("Fichier .clab trop volumineux pour un téléchargement via le navigateur : "
                   "récupère-le directement à l'emplacement indiqué ci-dessus.")
    st.write(f"Image reconstruite enregistrée : `{res['png_path'].relative_to(ROOT)}`")
    if res["png_path"].stat().st_size <= DOWNLOAD_MAX_BYTES:
        st.download_button("Télécharger l'image reconstruite", data=res["png_path"].read_bytes(),
                           file_name=res["png_path"].name, mime="image/png")

    st.subheader("8. Vérification sans perte")
    if res["exact"]:
        different_pixels, max_error = 0, 0
    else:
        different_pixels = int(np.count_nonzero((original_array != decoded_array).any(axis=2)))
        max_error = int(np.abs(original_array.astype(np.int16) - decoded_array.astype(np.int16)).max())
    equal_pixels = int(n_pixels - different_pixels)
    v1, v2, v3 = st.columns(3)
    v1.metric("Pixels identiques", f"{equal_pixels:,} / {n_pixels:,}")
    v2.metric("Pixels différents", f"{different_pixels:,}")
    v3.metric("Erreur maximale", str(max_error))
    if res["exact"] and different_pixels == 0 and max_error == 0:
        st.success("Compression sans perte confirmée : tous les pixels RGB sont identiques.")
    else:
        st.error("La reconstruction n'est pas identique à l'original.")


# ──────────────────────────────────────────────────────────────────────────────
# Mode « comparatif »
# ──────────────────────────────────────────────────────────────────────────────
def count_pipelines(n_stages, depth, n_coders):
    """Nombre de lignes du comparatif : chaque chaîne ordonnée, avec et sans codeur."""
    nodes = sum(perm(n_stages, k) for k in range(depth + 1))
    return nodes * (1 + n_coders)


def run_comparison(symbols, stage_names, max_depth, coder_names, progress=None):
    """Évalue toutes les chaînes ordonnées de préprocesseurs (jusqu'à max_depth),
    sans codeur ou avec Huffman / Arithmétique en terminal. Même logique que
    run_all_pipelines() de compress_lab.py, avec chronométrage et latence."""
    algorithms = get_algorithms()
    coders = get_coders()
    raw_bits = len(symbols) * 8
    n_pixels = max(len(symbols) // 3, 1)
    expected = count_pipelines(len(stage_names), max_depth, len(coder_names))
    results, skipped = [], {}

    def add(label, category, bits, enc_s, dec_s, ok):
        results.append({
            "Pipeline": label,
            "Catégorie": category,
            "Payload (bits)": int(bits),
            "Payload (octets)": (int(bits) + 7) // 8,
            "Gain (%)": gain_pct(raw_bits, bits),
            "Encodage (ms)": enc_s * 1000,
            "Décodage (ms)": dec_s * 1000,
            "Total (ms)": (enc_s + dec_s) * 1000,
            "Latence (µs/pixel)": (enc_s + dec_s) * 1e6 / n_pixels,
            "Sans perte": "OK" if ok else "ÉCHEC",
        })
        if progress:
            progress(min(len(results) / expected, 1.0))

    def decode_chain(stream, chain):
        cur = stream
        for name, meta in reversed(chain):
            cur = algorithms[name]["decode"](cur, meta)
        return cur

    def walk(seq, names, chain, plain_bits, enc_s):
        # 1) chaîne sans codeur entropique
        if names:
            started = time.perf_counter()
            ok = decode_chain(seq, chain) == symbols
            dec_s = time.perf_counter() - started
            category = "solo" if len(names) == 1 else "combiné"
            label = f"{names[0]} (solo)" if len(names) == 1 else "+".join(names) + " (sans entropie)"
            add(label, category, plain_bits, enc_s, dec_s, ok)
        else:
            add("Brut (référence)", "référence", raw_bits, 0.0, 0.0, True)

        # 2) chaîne + Huffman / Arithmétique
        for coder_name in coder_names:
            label = "+".join(names + [coder_name])
            try:
                check_limit(coder_name, len(seq))
                started = time.perf_counter()
                meta = coders[coder_name].enc(seq)
                coder_enc = time.perf_counter() - started
                started = time.perf_counter()
                restored = coders[coder_name].dec(meta)
                ok = (restored == seq) and decode_chain(restored, chain) == symbols
                dec_s = time.perf_counter() - started
                add(label, "entropie" if not names else "combiné", meta["payload"],
                    enc_s + coder_enc, dec_s, ok)
            except ValueError as exc:
                skipped[label] = str(exc)
                if progress:
                    progress(min((len(results) + len(skipped)) / expected, 1.0))

        # 3) descendre d'un niveau
        if len(names) >= max_depth:
            return
        for name in stage_names:
            if name in names or (name == "Delta" and names):
                continue  # Delta s'applique aux pixels bruts : première étape uniquement
            label = "+".join(names + [name])
            try:
                check_limit(name, len(seq))
                started = time.perf_counter()
                out, meta, bits = algorithms[name]["encode"](seq)
                step_s = time.perf_counter() - started
            except ValueError as exc:
                skipped[label] = str(exc)
                continue
            walk(out, names + [name], chain + [(name, meta)], bits, enc_s + step_s)

    walk(symbols, [], [], raw_bits, 0.0)
    df = pd.DataFrame(results).sort_values(["Payload (bits)", "Total (ms)"]).reset_index(drop=True)
    df.index += 1
    return df, skipped


def render_compare(res):
    """Affiche le tableau comparatif."""
    df = res["df"]
    st.subheader("4. Comparatif des pipelines")

    hide_negative = st.checkbox("Masquer les gains négatifs (comme la console)", value=True, key="hide_neg")
    shown = df[df["Gain (%)"] >= 0] if hide_negative else df
    hidden = len(df) - len(shown)

    candidates = df[(df["Catégorie"] != "référence") & (df["Sans perte"] == "OK")]
    best = candidates.iloc[0] if len(candidates) else None
    positive = candidates[candidates["Gain (%)"] > 0]
    fastest = positive.sort_values("Total (ms)").iloc[0] if len(positive) else None

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pipelines évalués", f"{len(df):,}", help=f"{hidden} masqué(s) (gain < 0)" if hide_negative else None)
    if best is not None:
        k2.metric("Meilleur gain", f"{best['Gain (%)']:.4f} %", delta=best["Pipeline"], delta_color="off")
    if fastest is not None:
        k3.metric("Plus rapide (gain > 0)", fmt_ms(fastest["Total (ms)"]), delta=fastest["Pipeline"], delta_color="off")
    k4.metric("Temps du comparatif", fmt_ms(res["elapsed_s"] * 1000))

    if not (df["Sans perte"] == "OK").all():
        st.error("Au moins une chaîne n'a pas pu être restituée sans perte (voir colonne « Sans perte »).")

    stretch(st.dataframe, shown, column_config={
        "Gain (%)": st.column_config.NumberColumn(format="%.4f"),
        "Encodage (ms)": st.column_config.NumberColumn(format="%.3f"),
        "Décodage (ms)": st.column_config.NumberColumn(format="%.3f"),
        "Total (ms)": st.column_config.NumberColumn(format="%.3f"),
        "Latence (µs/pixel)": st.column_config.NumberColumn(format="%.2f"),
    })
    st.caption(
        f"Référence : {res['n_symbols']:,} octets RGB bruts ({res['n_symbols'] * 8:,} bits), payload seul. "
        "Encodage/Décodage cumulent toutes les étapes de la chaîne. "
        "Latence = (encodage + décodage) / nombre de pixels."
    )

    top = shown.head(15)
    if len(top):
        st.markdown("**Top 15 — gain (%)**")
        stretch(st.bar_chart, top.set_index("Pipeline")["Gain (%)"])

    st.download_button("Télécharger le comparatif (CSV)",
                       data=df.to_csv(index_label="#").encode("utf-8"),
                       file_name=f"{res['stem']}_comparatif.csv", mime="text/csv")

    if res["skipped"]:
        with st.expander(f"{len(res['skipped'])} chaîne(s) ignorée(s) (limites de taille)"):
            for label, reason in res["skipped"].items():
                st.write(f"- `{label}` : {reason}")


# ──────────────────────────────────────────────────────────────────────────────
# Barre latérale
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")
st.sidebar.write("Les images doivent être placées dans `images/original/`.")

uploaded = st.sidebar.file_uploader(
    "Ou importer une image dans original/",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
)
if uploaded is not None:
    token = f"{uploaded.name}:{uploaded.size}"
    # N'écrire (et relancer) qu'une seule fois par fichier importé, sinon boucle infinie.
    if st.session_state.get("last_upload") != token:
        (ORIGINAL_DIR / Path(uploaded.name).name).write_bytes(uploaded.getvalue())
        st.session_state["last_upload"] = token
        st.sidebar.success(f"Image enregistrée : {uploaded.name}")
        st.rerun()

image_files = sorted(
    p for p in ORIGINAL_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
)

if not image_files:
    st.info(
        "Aucune image trouvée. Copie une image dans le dossier "
        "`images/original/`, puis actualise cette page."
    )
    st.code("mkdir -p images/original images/compressed\ncp ~/Images/mon_image.jpg images/original/")
    st.stop()

chosen_name = st.sidebar.selectbox("Image originale", [p.name for p in image_files])
selected_path = ORIGINAL_DIR / chosen_name

max_side_choice = st.sidebar.select_slider(
    "Taille de l'image testée (côté max, px)",
    options=RESIZE_CHOICES,
    value="Original",
    help="⚠ Réduire l'image n'est PAS sans perte : c'est seulement un moyen de tester vite. "
         "Pour compresser l'image entière sans perte, laisse « Original » et active le mode tuiles.",
)
max_side = 0 if max_side_choice == "Original" else int(max_side_choice)

st.sidebar.subheader("Algorithmes")
mode = st.sidebar.radio("Mode", ["Chaîne unique", "Comparatif automatique"])

if mode == "Chaîne unique":
    chosen_algos = st.sidebar.multiselect(
        "Algorithmes (ordre d'application)",
        STAGE_NAMES + CODER_NAMES,
        default=["Delta", "Huffman"],
        help="Choisis les préprocesseurs dans l'ordre voulu. Delta (prédiction) est toujours "
             "placé en premier. Huffman ou Arithmétique (un seul) sont des codeurs terminaux : "
             "toujours appliqués en dernier. Un codeur seul = mode « solo ».",
    )
    selected_stages = [a for a in chosen_algos if a in STAGE_NAMES]
    if "Delta" in selected_stages:
        selected_stages = ["Delta"] + [a for a in selected_stages if a != "Delta"]
    coders_selected = [a for a in chosen_algos if a in CODER_NAMES]
    entropy_name = coders_selected[0] if coders_selected else None
    config_error = "Choisis un seul codeur terminal (Huffman ou Arithmétique)." if len(coders_selected) > 1 else None

    use_tiles = st.sidebar.checkbox(
        "Compresser l'image entière par tuiles",
        value=False,
        help="Découpe l'image en blocs et compresse chacun indépendamment : permet de traiter "
             "une grande image SANS PERTE malgré les limites des algorithmes en Python pur. "
             "Les métadonnées sont répétées dans chaque tuile.",
    )
    tile_size = TILE_CHOICES[2]
    if use_tiles:
        tile_size = st.sidebar.select_slider("Taille des tuiles (px)", options=TILE_CHOICES, value=64)
        st.sidebar.caption(
            f"Une tuile de {tile_size}×{tile_size} contient {tile_size * tile_size * 3:,} valeurs "
            "(BWT : 5 000 max, Arithmétique : 30 000 max)."
        )
else:
    use_tiles, tile_size = False, 0
    compare_stages = st.sidebar.multiselect(
        "Préprocesseurs à combiner", STAGE_NAMES, default=["Delta", "RLE", "LZ77"])
    compare_depth = st.sidebar.slider("Longueur max des chaînes", 1, 4, 2)
    compare_coders = st.sidebar.multiselect(
        "Codeurs terminaux", CODER_NAMES, default=CODER_NAMES)
    config_error = None
    if compare_stages:
        est = count_pipelines(len(compare_stages), min(compare_depth, len(compare_stages)), len(compare_coders))
        st.sidebar.caption(f"≈ {est:,} pipelines à évaluer")
        if est > 300:
            st.sidebar.warning("Beaucoup de pipelines : le calcul peut être long. "
                               "Réduis la longueur ou l'image.")

matrix_limit = st.sidebar.slider("Taille de la matrice affichée", 3, 20, 8)

# ──────────────────────────────────────────────────────────────────────────────
# Image
# ──────────────────────────────────────────────────────────────────────────────
try:
    original_array, full_size, preview_array = load_array(
        str(selected_path), selected_path.stat().st_mtime, max_side)
except Exception as exc:
    st.error(f"Impossible de lire l'image : {exc}")
    st.stop()

height, width, channels = original_array.shape
full_w, full_h = full_size
resized = (width, height) != (full_w, full_h)
original_file_size = selected_path.stat().st_size
raw_pixel_bytes = original_array.nbytes

st.subheader("1. Image originale")
left, right = st.columns([1, 1])
with left:
    stretch(st.image, preview_array, caption=f"Originale : {chosen_name}")
with right:
    st.markdown("**Informations de l'image**")
    st.write(f"- Dimensions du fichier : **{full_w} × {full_h} pixels**")
    if resized:
        st.write(f"- Dimensions testées (réduites) : **{width} × {height} pixels**")
    st.write(f"- Canaux : **{channels} (RGB)**")
    st.write(f"- Taille du fichier original : **{original_file_size:,} octets**")
    st.write(f"- Données brutes des pixels testés : **{raw_pixel_bytes:,} octets**")
    st.write(f"- Type de matrice : **{original_array.dtype}**")
    if original_array.size > MAX_SYMBOLS and not (mode == "Chaîne unique" and use_tiles):
        st.warning(
            f"Cette image contient {original_array.size:,} valeurs RGB (maximum {MAX_SYMBOLS:,} "
            "d'un seul bloc). Pour la compresser entière et sans perte, active "
            "« Compresser l'image entière par tuiles » dans la barre latérale. "
            "(Réduire l'image permet seulement de tester vite, mais n'est pas sans perte.)"
        )

st.subheader("2. Matrice des pixels originaux")
st.caption(f"Extrait {matrix_limit} × {matrix_limit} pixels — chaque pixel contient [R, G, B].")
stretch(st.dataframe, original_array[:matrix_limit, :matrix_limit, :].tolist())

# ──────────────────────────────────────────────────────────────────────────────
# Compression
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("3. Compression et décompression")

if mode == "Chaîne unique":
    chain = selected_stages + ([entropy_name] if entropy_name else [])
    signature = ("single", chosen_name, selected_path.stat().st_mtime, max_side, tuple(chain),
                 use_tiles, tile_size)
    st.write("Chaîne choisie :", " → ".join(chain) if chain else "Aucun algorithme (pixels bruts)")
    if use_tiles:
        n_tiles_est = -(-height // tile_size) * -(-width // tile_size)
        st.write(f"Mode tuiles : **{n_tiles_est:,} tuiles** de {tile_size} × {tile_size} pixels "
                 "(image entière, sans perte).")
else:
    chain = []
    signature = ("compare", chosen_name, selected_path.stat().st_mtime, max_side,
                 tuple(compare_stages), compare_depth, tuple(compare_coders))
    st.write("Préprocesseurs :", ", ".join(compare_stages) or "—",
             "· codeurs :", ", ".join(compare_coders) or "aucun",
             f"· longueur max : {compare_depth}")

label_btn = "🚀 Lancer le test" if mode == "Chaîne unique" else "🚀 Lancer le comparatif"
run_clicked = stretch(st.button, label_btn, type="primary")

if run_clicked:
    st.session_state.pop("result", None)
    try:
        n_values = int(original_array.size)
        if mode == "Chaîne unique" and not use_tiles and n_values > MAX_SYMBOLS:
            raise ValueError(
                f"Image trop grande pour être traitée d'un seul bloc ({n_values:,} valeurs RGB, "
                f"maximum {MAX_SYMBOLS:,}). Active « Compresser l'image entière par tuiles » "
                "pour la compresser sans perte, ou réduis-la avec « Taille de l'image testée » "
                "(test seulement : réduire n'est pas sans perte)."
            )
        if config_error:
            raise ValueError(config_error)

        if mode == "Chaîne unique":
            slug = "+".join(chain).replace("é", "e") or "brut"
            suffix = f"_tuiles{tile_size}" if use_tiles else ""
            png_path = COMPRESSED_DIR / f"{selected_path.stem}_reconstructed.png"
            clab_path = COMPRESSED_DIR / f"{selected_path.stem}_{slug}{suffix}.clab"
            started = time.perf_counter()

            if use_tiles:
                bar = st.progress(0.0, text="Compression par tuiles…")

                def on_progress(done, total, elapsed, eta):
                    bar.progress(done / total,
                                 text=f"Tuile {done:,} / {total:,} — écoulé {elapsed:.0f} s, restant ≈ {eta:.0f} s")

                res = compress_tiled(original_array, selected_stages, entropy_name,
                                     tile_size, clab_path, progress=on_progress)
                bar.progress(1.0, text="Enregistrement de l'image reconstruite…")
            else:
                with st.spinner("Compression puis décompression en cours..."):
                    res = compress_pixels(original_array, selected_stages, entropy_name)
                clab_path.write_bytes(res["blob"])

            reconstructed = Image.fromarray(res["decoded_array"])
            reconstructed.save(png_path, format="PNG", compress_level=1)
            res["total_s"] = time.perf_counter() - started
            if use_tiles:
                bar.empty()

            preview_rebuilt = reconstructed.copy()
            preview_rebuilt.thumbnail((1200, 1200), Image.LANCZOS)
            res.update({
                "original_array": original_array,
                "preview_original": preview_array,
                "preview_rebuilt": np.asarray(preview_rebuilt),
                "png_path": png_path,
                "clab_path": clab_path,
                "matrix_limit": matrix_limit,
                "resized": resized,
                "original_file_size": original_file_size,
            })
            st.session_state["result"] = {"signature": signature, "kind": "single", "data": res}
        else:
            if not compare_stages:
                raise ValueError("Choisis au moins un préprocesseur à combiner.")
            symbols = original_array.reshape(-1).tolist()
            bar = st.progress(0.0, text="Évaluation des pipelines…")
            started = time.perf_counter()
            df, skipped = run_comparison(
                symbols, compare_stages, min(compare_depth, len(compare_stages)),
                compare_coders, progress=lambda p: bar.progress(p, text="Évaluation des pipelines…"))
            elapsed = time.perf_counter() - started
            bar.empty()
            st.session_state["result"] = {"signature": signature, "kind": "compare", "data": {
                "df": df, "skipped": skipped, "elapsed_s": elapsed,
                "n_symbols": len(symbols), "stem": selected_path.stem,
            }}
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.exception(exc)

# Les résultats sont conservés en session : ils ne disparaissent plus quand on
# touche à une case à cocher ou à un bouton de téléchargement.
stored = st.session_state.get("result")
if stored is not None:
    if stored["signature"] == signature:
        if stored["kind"] == "single":
            render_single(stored["data"])
        else:
            render_compare(stored["data"])
    else:
        st.info("La configuration a changé depuis le dernier test : relance-le pour mettre les résultats à jour.")

st.divider()
st.caption("Image Compression Lab — réutilise directement les fonctions de compress_lab.py.")
