#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Laboratoire de compression d'images utilisant les algorithmes de compress_lab.py.
Placer ce fichier dans le même dossier que compress_lab.py.

Deux modes :
  • Chaîne unique   : Delta / RLE / BWT / LZ77 / LZ78 / LZW, suivis (ou non) de Huffman
                      ou de l'Arithmétique (codeur terminal), avec mesures détaillées.
                      Option « tuiles » : compresse l'image ENTIÈRE, sans perte, bloc par bloc.
  • Comparatif      : classe rapidement les chaînes (solo, combinées, avec
                      Huffman / Arithmétique) sur un échantillon borné, avec
                      recherche guidée et vérification réelle des meilleurs
                      candidats. Le mode exhaustif reste disponible.

Mesures : gain, RC, temps d'encodage / décodage / total, latence, débit,
          et qualité de reconstruction (MSE, SNR, PSNR, SSIM) avec explications pédagogiques.
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
    "BWT": 64_000,           # suffix-array cyclique par doublement
    "LZ77": 100_000,
    "LZ78": 100_000,
    "LZW": 100_000,
    "Huffman": 1_000_000,
    "Arithmétique": 30_000,  # entiers de plusieurs millions de bits
}

RESIZE_CHOICES = ["Original", 32, 64, 128, 256, 512, 1024]
TILE_CHOICES = [16, 32, 64, 128]
DOWNLOAD_MAX_BYTES = 200_000_000

# Comparatif : on n'a pas besoin de compresser toute l'image pour classer les
# pipelines. Un échantillon réparti dans l'image donne le même classement en
# quelques centaines de millisecondes, même sur une photo de plusieurs mégapixels.
COMPARE_SAMPLE_CHOICES = [5_000, 10_000, 20_000, 40_000, 64_000]
COMPARE_DEFAULT_SAMPLE = 10_000
COMPARE_DEFAULT_BEAM = 16
COMPARE_VERIFY_TOP = 6
COMPARE_ARITH_VERIFY_MAX = 15_000  # au-delà, on garde l'estimation (codage réel trop lent)

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


def fmt_duration(seconds):
    """Durée lisible : millisecondes, secondes, minutes ou heures."""
    seconds = max(float(seconds), 0.0)
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 10:
        return f"{seconds:.1f} s"
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def fmt_bytes(n):
    """Affiche une taille en octets de façon lisible (Ko, Mo…)."""
    n = float(n)
    for unit, div in (("Mo", 1_048_576), ("Ko", 1024)):
        if abs(n) >= div:
            return f"{n / div:,.2f} {unit}"
    return f"{int(n):,} octets"


def compute_quality_metrics(original, reconstructed):
    """Calcule MSE, SNR, PSNR et SSIM entre deux images RGB (uint8).

    Ces métriques mesurent la *fidélité* de la reconstruction :
    - MSE  : erreur quadratique moyenne (0 = identique).
    - SNR  : rapport signal/bruit en dB (plus élevé = mieux).
    - PSNR : peak signal-to-noise ratio en dB (plus élevé = mieux ; ∞ si MSE=0).
    - SSIM : similarité structurelle (1 = identique, 0 = totalement différent).

    Pour une compression *sans perte*, on attend MSE=0, PSNR=∞, SSIM=1.
    """
    if original.shape != reconstructed.shape:
        raise ValueError("Les images à comparer n'ont pas la même forme.")

    # Cas normal de l'application : reconstruction exacte. On évite alors les
    # tableaux float64 (plusieurs Go sur une grande image) et le calcul SSIM.
    if original is reconstructed or np.array_equal(original, reconstructed):
        return {
            "mse": 0.0,
            "snr_db": float("inf"),
            "psnr_db": float("inf"),
            "ssim": 1.0,
        }

    mse, signal_power = _mse_and_signal_power(original, reconstructed)
    if mse < 1e-12:
        snr_db = float("inf")
        psnr_db = float("inf")
    else:
        snr_db = 10.0 * np.log10(signal_power / mse) if signal_power > 0 else float("-inf")
        psnr_db = 10.0 * np.log10((255.0 ** 2) / mse)

    ssim_val = _ssim_rgb_limited(original, reconstructed)
    return {
        "mse": mse,
        "snr_db": snr_db,
        "psnr_db": psnr_db,
        "ssim": ssim_val,
    }


def _mse_and_signal_power(original, reconstructed, target_values=1_000_000):
    """Calcule MSE et puissance du signal par bandes de lignes (RAM bornée)."""
    height, width, channels = original.shape
    rows = max(1, int(target_values // max(width * channels, 1)))
    mse_sum = 0.0
    signal_sum = 0.0
    count = 0
    for y in range(0, height, rows):
        o = original[y:y + rows].astype(np.float64)
        r = reconstructed[y:y + rows].astype(np.float64)
        d = o - r
        mse_sum += float(np.sum(d * d))
        signal_sum += float(np.sum(o * o))
        count += int(o.size)
    if not count:
        return 0.0, 0.0
    return mse_sum / count, signal_sum / count


def _ssim_rgb_limited(original, reconstructed, max_values=4_000_000, max_side=1024):
    """SSIM RGB ; réduit l'image si elle est trop grande pour rester en RAM."""
    if original.size <= max_values:
        return _ssim_rgb(original.astype(np.float64),
                         reconstructed.astype(np.float64))
    height, width = original.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small_original = np.asarray(
        Image.fromarray(original).resize(size, Image.LANCZOS), dtype=np.float64)
    small_reconstructed = np.asarray(
        Image.fromarray(reconstructed).resize(size, Image.LANCZOS), dtype=np.float64)
    return _ssim_rgb(small_original, small_reconstructed)


def _ssim_rgb(img1, img2, sigma=1.5, K1=0.01, K2=0.03, L=255.0):
    """SSIM moyen sur les 3 canaux RGB (filtre gaussien)."""
    from scipy import ndimage

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    scores = []
    for c in range(img1.shape[2]):
        x = img1[:, :, c]
        y = img2[:, :, c]
        mu1 = ndimage.gaussian_filter(x, sigma=sigma, truncate=3.5)
        mu2 = ndimage.gaussian_filter(y, sigma=sigma, truncate=3.5)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = ndimage.gaussian_filter(x * x, sigma=sigma, truncate=3.5) - mu1_sq
        sigma2_sq = ndimage.gaussian_filter(y * y, sigma=sigma, truncate=3.5) - mu2_sq
        sigma12 = ndimage.gaussian_filter(x * y, sigma=sigma, truncate=3.5) - mu1_mu2
        sigma1_sq = np.maximum(sigma1_sq, 0)
        sigma2_sq = np.maximum(sigma2_sq, 0)
        num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = num / den
        scores.append(float(np.mean(ssim_map)))
    return float(np.mean(scores))


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
                if not r["exact"]:
                    raise ValueError(
                        f"La tuile ({ty}, {tx}) n'a pas été reconstruite exactement.")
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

    # Toutes les tuiles sont vérifiées exactes : l'image reconstruite est
    # identique à l'originale. On réutilise le tableau d'origine au lieu d'en
    # garder une copie complète (plusieurs centaines de Mo sur une grande image).
    if not exact_all:
        raise ValueError("Une ou plusieurs tuiles n'ont pas été reconstruites exactement.")
    decoded = pixel_array
    return {
        "decoded_array": decoded,
        "rows": agg_rows or [],
        "raw_bits": raw_bits,
        "payload_bits": payload_bits,
        "exact": True,
        "blob": None,
        "blob_size": clab_path.stat().st_size,
        "tiles": {"size": tile, "count": n_tiles, "gain_min": gain_min,
                  "gain_max": gain_max, "expanded": expanded},
        **sums,
    }


@st.cache_data(show_spinner=False)
def benchmark_tile(tile_bytes, shape, stages, entropy_name):
    """Benchmarke une tuile réelle pour estimer la durée du mode tuiles.

    Le benchmark fait le même travail qu'une tuile réelle (encodage,
    sérialisation, décodage, vérification). Le résultat est mis en cache par
    configuration pour ne pas ralentir les interactions suivantes.
    """
    block = np.frombuffer(tile_bytes, dtype=np.uint8).reshape(shape)
    started = time.perf_counter()
    try:
        res = compress_pixels(block, list(stages), entropy_name)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "per_tile_s": time.perf_counter() - started,
        "raw_bits": res["raw_bits"],
        "payload_bits": res["payload_bits"],
        "blob_size": res["blob_size"],
        "exact": res["exact"],
    }


def representative_tile(pixel_array, tile_size):
    """Tuile centrale (ou image entière si elle est plus petite qu'une tuile)."""
    height, width, _ = pixel_array.shape
    ty = max(0, (height - tile_size) // 2)
    tx = max(0, (width - tile_size) // 2)
    return np.ascontiguousarray(pixel_array[ty:ty + tile_size, tx:tx + tile_size])


def representative_tiles(pixel_array, tile_size, count=3):
    """Quelques tuiles réparties dans l'image (centre, quart, trois-quarts)."""
    if count <= 1:
        return [representative_tile(pixel_array, tile_size)]
    height, width, _ = pixel_array.shape
    positions = ((0.5, 0.5), (0.25, 0.25), (0.75, 0.75))
    tiles = []
    for fy, fx in positions[:count]:
        ty = min(max(0, int(height * fy - tile_size / 2)), max(0, height - tile_size))
        tx = min(max(0, int(width * fx - tile_size / 2)), max(0, width - tile_size))
        tiles.append(np.ascontiguousarray(
            pixel_array[ty:ty + tile_size, tx:tx + tile_size]))
    return tiles


def render_single(res):
    """Affiche les résultats d'une exécution en chaîne unique."""
    original_array = res["original_array"]
    decoded_array = res["decoded_array"]
    height, width, _ = original_array.shape
    n_pixels = height * width
    raw_bytes = original_array.nbytes
    payload_bits = res["payload_bits"]
    payload_bytes = (payload_bits + 7) // 8
    ratio = (raw_bytes / payload_bytes) if payload_bytes else 0.0
    blob_size = res["blob_size"]

    # Métriques de qualité (MSE / SNR / PSNR / SSIM)
    quality = compute_quality_metrics(original_array, decoded_array)

    st.subheader("4. Résultat visuel")
    col1, col2 = st.columns(2)
    with col1:
        stretch(st.image, res["preview_original"], caption="Image testée (entrée)")
    with col2:
        stretch(st.image, res["preview_rebuilt"], caption="Image reconstruite (sortie)")

    if res["exact"]:
        st.success("✅ Vérification sans perte réussie : les matrices de pixels sont **strictement identiques**.")
    else:
        st.error("❌ La reconstruction diffère de l'image originale (compression avec perte ou erreur).")

    # ── Métriques clés ───────────────────────────────────────────────────────
    st.subheader("5. Métriques clés (MSE · SSIM · RC · SNR · PSNR)")

    mse_txt = f"{quality['mse']:.6f}" if quality["mse"] > 0 else "0"
    ssim_txt = f"{quality['ssim']:.6f}"
    rc_txt = f"{ratio:.2f} : 1"
    snr_txt = "∞ dB" if quality["snr_db"] == float("inf") else f"{quality['snr_db']:.2f} dB"
    psnr_txt = "∞ dB" if quality["psnr_db"] == float("inf") else f"{quality['psnr_db']:.2f} dB"

    with st.container(border=True):
        q1, q2 = st.columns(2)
        with q1:
            st.markdown(f"**MSE** = {mse_txt}")
            st.markdown(f"**SSIM** = {ssim_txt}")
            st.markdown(f"**RC** = {rc_txt}")
        with q2:
            st.markdown(f"**SNR** = {snr_txt}")
            st.markdown(f"**PSNR** = {psnr_txt}")

    st.subheader("6. Matrice reconstruite (aperçu)")
    m = res["matrix_limit"]
    st.caption(f"Extrait {m}×{m} pixels — chaque case = [R, G, B]. Utile pour vérifier visuellement les valeurs.")
    stretch(st.dataframe, decoded_array[:m, :m, :].tolist())

    st.subheader("7. Détails des étapes de la chaîne")
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
        with st.expander("📖 Comment lire ce tableau ? (débutants)"):
            st.markdown(
                """
- **Symboles avant / après** : nombre de valeurs qui entrent et sortent de l’étape.
  Une réduction forte (ex. RLE sur une image unie) est un bon signe.
- **Payload (bits)** : estimation de la taille des données *après* cette étape
  (sans compter encore les tables de Huffman, etc.).
- **Gain cumulé (%)** : `(taille_brute − payload) / taille_brute × 100`.
  - Positif → on a gagné de la place.
  - Négatif → l’étape a *agrandi* les données (fréquent si on applique un algo inadapté).
- **Latence** : temps moyen passé par symbole. Utile pour comparer la complexité des algos.
- **Vérification** : « OK » signifie que le décodage de cette étape a parfaitement
  retrouvé le flux d’entrée (indispensable pour du sans-perte).
"""
            )
    else:
        st.info("Aucun algorithme sélectionné : les pixels sont stockés tels quels.")

    st.subheader("8. Mesures de compression (tailles & gains)")

    # ---- Tailles et gain
    saved_payload = raw_bytes - payload_bytes
    gain_payload = gain_pct(res["raw_bits"], payload_bits)
    gain_container = (raw_bytes - blob_size) / raw_bytes * 100 if raw_bytes else 0.0
    saved_container = raw_bytes - blob_size

    st.markdown("#### 📦 Taille des données")
    st.markdown(
        f"""
| Notion | Valeur | Explication simple |
|--------|--------|--------------------|
| **Pixels RGB bruts** | **{raw_bytes:,} octets** ({fmt_bytes(raw_bytes)}) | Largeur × hauteur × 3 canaux × 1 octet. C’est la référence « non compressée ». |
| **Payload compressé** | **{payload_bytes:,} octets** ({fmt_bytes(payload_bytes)}) | Taille des *seules données* compressées (estimation bit-exacte). |
| **Conteneur .clab** | **{blob_size:,} octets** ({fmt_bytes(blob_size)}) | Fichier réel écrit sur disque (payload + tables Huffman/Arith + métadonnées + pickle). |
"""
    )

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Pixels RGB bruts", f"{raw_bytes:,} o", help="Taille de référence non compressée.")
    a2.metric("Payload compressé", f"{payload_bytes:,} o",
              help="Données compressées uniquement (sans tables ni en-têtes).")
    a3.metric("Gain (payload)", f"{gain_payload:.4f} %",
              delta=f"{saved_payload:+,} o",
              help="Pourcentage de place gagnée sur le payload par rapport aux pixels bruts.")
    a4.metric("RC (payload)", f"{ratio:.2f} : 1",
              help="Rapport de compression = pixels bruts / payload compressé.")

    b1, b2, b3 = st.columns(3)
    b1.metric("Conteneur .clab réel", f"{blob_size:,} o")
    b2.metric("Gain (conteneur vs pixels bruts)", f"{gain_container:.4f} %",
              delta=f"{saved_container:+,} o")
    if res["resized"]:
        b3.metric("Gain vs fichier original", "—",
                  help="Non comparable : l'image a été réduite avant le test.")
    else:
        orig_size = res["original_file_size"]
        gain_vs_file = (orig_size - blob_size) / orig_size * 100 if orig_size else 0.0
        b3.metric("Gain (conteneur vs fichier original)", f"{gain_vs_file:.4f} %",
                  delta=f"{orig_size - blob_size:+,} o",
                  help="Compare le .clab au fichier image d’origine (PNG/JPEG… déjà compressé).")

    with st.expander("📖 Comprendre les gains et les tailles (débutants)"):
        st.markdown(
            f"""
**1. Pixels RGB bruts**  
C’est la taille « théorique » de l’image non compressée :  
`largeur × hauteur × 3` octets.  
Exemple : 512×512 → 786 432 octets.

**2. Payload**  
Taille des données *après* les algorithmes (Delta, RLE, LZ… + Huffman/Arith).  
C’est ce que la console de `compress_lab` affiche.  
Gain payload = `(bruts − payload) / bruts × 100`.

**3. Conteneur .clab**  
Le fichier réellement écrit. Il contient :
- le payload,
- les tables de Huffman (ou paramètres arithmétiques),
- la forme de l’image, la liste des étapes, etc.,
- le format pickle (pédagogique, pas un standard industriel).

Le conteneur est **toujours un peu plus gros** que le payload.  
Si l’écart est très grand (> 15 points de %), les métadonnées pèsent trop
(tuiles trop petites, ou beaucoup de symboles distincts pour Huffman).

**4. Gain vs fichier original**  
Le fichier PNG/JPEG d’origine est *déjà* compressé.  
Comparer le .clab à ce fichier montre si notre chaîne pédagogique
fait mieux ou moins bien qu’un format réel.

**Interprétation rapide de ton résultat actuel :**
- Gain payload ≈ **{gain_payload:.2f} %**
- Gain conteneur ≈ **{gain_container:.2f} %**
- RC ≈ **{ratio:.2f}:1**
"""
        )

    # ---- Temps et latence
    st.markdown("#### ⏱️ Temps, latence et débit")
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
        st.markdown("#### 🧩 Mode tuiles")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Tuiles", f"{t['count']:,}", help=f"Tuiles de {t['size']} × {t['size']} pixels")
        e2.metric("Gain min. d'une tuile", f"{t['gain_min']:.2f} %")
        e3.metric("Gain max. d'une tuile", f"{t['gain_max']:.2f} %")
        e4.metric("Tuiles en expansion", f"{t['expanded']:,}",
                  help="Tuiles dont le payload dépasse les pixels bruts.")

    if gain_payload < 0:
        st.warning("⚠️ Le payload est plus volumineux que les pixels bruts : cette combinaison "
                   "n'est pas adaptée à cette image (gain négatif).")
    if gain_container < 0:
        st.warning("⚠️ Le conteneur .clab est plus volumineux que les pixels bruts.")
    elif gain_payload - gain_container > 15:
        st.warning(
            f"⚠️ Le conteneur réel gagne beaucoup moins que le payload ({gain_container:.1f} % contre "
            f"{gain_payload:.1f} %) : les tables et métadonnées pèsent lourd. C'est fréquent avec des "
            "tuiles trop petites, ou quand Huffman code des jetons LZ77/LZ78 (milliers de symboles "
            "distincts). Essaie des tuiles plus grandes ou la chaîne **Delta → Huffman**."
        )

    st.caption(
        "Le **gain (payload)** suit la méthode de la console : taille des seules données compressées, "
        "sans les métadonnées. Le **conteneur .clab** inclut en plus les tables et métadonnées nécessaires "
        "à la reconstruction ; il est sérialisé avec pickle pour ce laboratoire pédagogique (ce n'est pas "
        "un format universel). Sans codeur Huffman/Arithmétique, le flux intermédiaire est stocké sous forme "
        "d'objets Python, d'où un conteneur nettement plus gros que le payload estimé. "
        "**N'ouvre jamais un fichier pickle provenant d'une source inconnue.**"
    )

    st.write(f"📁 Flux compressé enregistré : `{res['clab_path'].relative_to(ROOT)}`")
    if res["blob"] is not None:
        st.download_button("⬇️ Télécharger le flux compressé (.clab)", data=res["blob"],
                           file_name=res["clab_path"].name, mime="application/octet-stream")
    elif blob_size <= DOWNLOAD_MAX_BYTES:
        st.download_button("⬇️ Télécharger le flux compressé (.clab)", data=res["clab_path"].read_bytes(),
                           file_name=res["clab_path"].name, mime="application/octet-stream")
    else:
        st.caption("Fichier .clab trop volumineux pour un téléchargement via le navigateur : "
                   "récupère-le directement à l'emplacement indiqué ci-dessus.")
    st.write(f"🖼️ Image reconstruite enregistrée : `{res['png_path'].relative_to(ROOT)}`")
    if res["png_path"].stat().st_size <= DOWNLOAD_MAX_BYTES:
        st.download_button("⬇️ Télécharger l'image reconstruite", data=res["png_path"].read_bytes(),
                           file_name=res["png_path"].name, mime="image/png")

    st.subheader("9. Vérification pixel par pixel (sans perte)")
    if res["exact"]:
        different_pixels, max_error = 0, 0
    else:
        different_pixels = int(np.count_nonzero((original_array != decoded_array).any(axis=2)))
        max_error = int(np.abs(original_array.astype(np.int16) - decoded_array.astype(np.int16)).max())
    equal_pixels = int(n_pixels - different_pixels)
    v1, v2, v3 = st.columns(3)
    v1.metric("Pixels identiques", f"{equal_pixels:,} / {n_pixels:,}")
    v2.metric("Pixels différents", f"{different_pixels:,}")
    v3.metric("Erreur maximale (canal)", str(max_error),
              help="Différence absolue maximale sur un canal R, G ou B.")

    if res["exact"] and different_pixels == 0 and max_error == 0:
        st.success("✅ Compression **sans perte** confirmée : tous les pixels RGB sont strictement identiques.")
    else:
        st.error("❌ La reconstruction n'est pas identique à l'original.")


# ──────────────────────────────────────────────────────────────────────────────
# Mode « comparatif »
# ──────────────────────────────────────────────────────────────────────────────
def estimate_pipeline_rows(stage_names, depth, n_coders, guided=True,
                           beam_width=COMPARE_DEFAULT_BEAM):
    """Nombre approximatif de lignes du comparatif.

    On tient compte de deux réductions essentielles :
    • Delta n'est autorisé qu'en première position ;
    • en recherche guidée, seules les meilleures branches sont prolongées
      au-delà de deux étapes (toutes les chaînes de 1–2 étapes restent testées).
    """
    names = [n for n in stage_names if n != "Delta"]
    m, has_delta = len(names), "Delta" in stage_names
    total, frontier = 1, 1
    for d in range(1, depth + 1):
        exact = perm(m, d) + (perm(m, d - 1) if has_delta else 0)
        if not guided or d <= 2:
            candidates = exact
        else:
            remaining = max(m - (d - 1) + (1 if has_delta else 0), 0)
            candidates = min(exact, frontier * remaining)
        total += candidates
        frontier = min(candidates, beam_width) if guided and d >= 2 else candidates
    return total * (1 + n_coders)


def sample_flat_array(flat, limit, n_chunks=8):
    """Échantillon représentatif de `limit` valeurs RGB, réparti en blocs.

    On garde des blocs contigus (alignés sur les 3 canaux) : Delta conserve
    ainsi les corrélations locales, tout en couvrant toute l'image au lieu de
    se limiter au coin supérieur gauche. Retourne (échantillon, blocs, total).
    """
    total = int(flat.size)
    if not limit or int(limit) >= total:
        return np.asarray(flat), 1, total
    limit = min(int(limit), total)
    limit -= limit % 3
    if limit < 3:
        return np.asarray(flat[:limit]), 1, total
    n_chunks = max(1, min(int(n_chunks), limit // 3, total // 3))
    chunk = (limit // n_chunks) // 3 * 3
    if chunk < 3:
        return np.asarray(flat[:limit]), 1, total
    stride = max(chunk, ((total // n_chunks) // 3) * 3)
    parts = []
    for c in range(n_chunks):
        start = c * stride
        if start + chunk > total:
            start = max(0, total - chunk)
        start = (start // 3) * 3
        parts.append(flat[start:start + chunk])
    sample = np.concatenate(parts) if len(parts) > 1 else parts[0]
    return sample, len(parts), total


def run_comparison(symbols, stage_names, max_depth, coder_names, progress=None,
                   guided=True, beam_width=COMPARE_DEFAULT_BEAM,
                   verify_top=COMPARE_VERIFY_TOP):
    """Compare les chaînes de préprocesseurs de façon rapide et fiable.

    Démarche :
    1. Toutes les chaînes de 1 et 2 étapes sont évaluées (peu nombreuses).
    2. Au-delà, une recherche guidée ne prolonge que les meilleures branches
       (faisceau), en gardant au moins une branche par dernière technique.
    3. Les payloads Huffman sont exacts ; ceux de l'Arithmétique sont estimés
       par l'entropie. Le vrai codage arithmétique, très coûteux, est réservé
       aux meilleurs candidats.
    4. Les meilleures chaînes sont réencodées/décodées avec le vrai codeur
       pour confirmer le sans-perte et mesurer des temps réels.
    """
    algorithms = get_algorithms()
    coders = get_coders()
    raw_bits = len(symbols) * 8
    n_pixels = max(len(symbols) // 3, 1)
    expected = estimate_pipeline_rows(stage_names, max_depth, len(coder_names),
                                      guided, beam_width)
    results, skipped = [], {}
    progress_rows = [0]
    last_progress = [0.0]
    verified_labels = set()

    def push_progress(frac):
        """Limite les mises à jour Streamlit à ~1 % de progression."""
        frac = min(max(float(frac), 0.0), 1.0)
        if progress and (frac >= 1.0 or frac - last_progress[0] >= 0.01):
            last_progress[0] = frac
            progress(frac)

    def add_row(label, category, bits, enc_s, dec_s, ok, coder_name, names,
                verified):
        total_s = None if enc_s is None or dec_s is None else enc_s + dec_s
        status = "OK" if ok else "ÉCHEC"
        if coder_name and not verified:
            status = "OK (estimé)" if ok else "ÉCHEC"
        results.append({
            "Pipeline": label,
            "Catégorie": category,
            "Payload (bits)": int(bits),
            "Payload (octets)": (int(bits) + 7) // 8,
            "Gain (%)": gain_pct(raw_bits, bits),
            "RC (×)": (raw_bits / bits) if bits else 0.0,
            "Encodage (ms)": None if enc_s is None else enc_s * 1000,
            "Décodage (ms)": None if dec_s is None else dec_s * 1000,
            "Total (ms)": None if total_s is None else total_s * 1000,
            "Latence (µs/pixel)": None if total_s is None else total_s * 1e6 / n_pixels,
            "Sans perte": status,
            "Mesure": "réelle" if verified else "estimée",
            "_names": tuple(names),
            "_coder": coder_name,
        })
        progress_rows[0] += 1
        push_progress(min(0.88 * progress_rows[0] / max(expected, 1), 0.88))

    def label_for(names, coder_name=None):
        base = "+".join(names)
        if coder_name:
            return f"{base}+{coder_name}" if base else coder_name
        if not names:
            return "Brut (référence)"
        if len(names) == 1:
            return ("RLE classique (solo)" if names[0] == "RLE"
                    else f"{names[0]} (solo)")
        return f"{base} (sans entropie)"

    def add_variants(state):
        names, seq = state["names"], state["seq"]
        if names:
            add_row(label_for(names), "solo" if len(names) == 1 else "combiné",
                    state["plain_bits"], state["enc_s"], state["dec_s"],
                    state["ok"], None, names, True)
        else:
            add_row("Brut (référence)", "référence", raw_bits, 0.0, 0.0,
                    True, None, names, True)

        scores = [int(state["plain_bits"])]
        for coder_name in coder_names:
            if coder_name == "Huffman":
                bits = lab.Huffman.payload_estimate(seq)
            elif coder_name == "Arithmétique":
                bits = lab.Arith.payload_estimate(seq)
            else:
                continue
            add_row(label_for(names, coder_name),
                    "entropie" if not names else "combiné",
                    bits, None, None, state["ok"], coder_name, names, False)
            scores.append(int(bits))
        return min(scores)

    def expand(state, stage_name):
        seq = state["seq"]
        names = state["names"] + [stage_name]
        path = "+".join(names)
        try:
            check_limit(stage_name, len(seq))
            started = time.perf_counter()
            out, meta, bits = algorithms[stage_name]["encode"](seq)
            enc_s = time.perf_counter() - started
            started = time.perf_counter()
            restored = algorithms[stage_name]["decode"](out, meta)
            dec_s = time.perf_counter() - started
            ok = restored == seq
        except ValueError as exc:
            skipped[path] = str(exc)
            return None
        except Exception as exc:
            skipped[path] = f"{type(exc).__name__}: {exc}"
            return None
        return {
            "seq": out,
            "names": names,
            "plain_bits": int(bits),
            "enc_s": state["enc_s"] + enc_s,
            "dec_s": state["dec_s"] + dec_s,
            "ok": state["ok"] and ok,
            "last": stage_name,
        }

    def select_frontier(states, width):
        """Garde les meilleures branches, avec au moins une par dernière étape."""
        if len(states) <= width:
            return states
        ranked = sorted(states, key=lambda s: (s["score"], s["plain_bits"]))
        chosen, chosen_idx, seen_last = [], set(), set()
        for i, state in enumerate(ranked):
            if state["last"] not in seen_last:
                chosen.append(state)
                chosen_idx.add(i)
                seen_last.add(state["last"])
        for i, state in enumerate(ranked):
            if len(chosen) >= width:
                break
            if i not in chosen_idx:
                chosen.append(state)
        return chosen[:width]

    def verify_row(row):
        """Réencode réellement une chaîne et met à jour sa ligne."""
        names, coder_name = list(row["_names"]), row["_coder"]
        seq = list(symbols)
        chain, enc_s, dec_s = [], 0.0, 0.0
        payload = int(row["Payload (bits)"])
        ok = True
        try:
            for stage_name in names:
                check_limit(stage_name, len(seq))
                started = time.perf_counter()
                out, meta, bits = algorithms[stage_name]["encode"](seq)
                enc_s += time.perf_counter() - started
                chain.append((stage_name, meta))
                seq, payload = out, int(bits)
            final = seq
            if coder_name:
                check_limit(coder_name, len(seq))
                started = time.perf_counter()
                meta = coders[coder_name].enc(seq)
                enc_s += time.perf_counter() - started
                payload = int(meta["payload"])
                started = time.perf_counter()
                final = coders[coder_name].dec(meta)
                dec_s += time.perf_counter() - started
                ok = final == seq
            for stage_name, meta in reversed(chain):
                started = time.perf_counter()
                final = algorithms[stage_name]["decode"](final, meta)
                dec_s += time.perf_counter() - started
            ok = ok and final == symbols
        except ValueError as exc:
            skipped[row["Pipeline"]] = str(exc)
            return False
        except Exception as exc:
            skipped[row["Pipeline"]] = f"{type(exc).__name__}: {exc}"
            return False

        total_s = enc_s + dec_s
        row.update({
            "Payload (bits)": payload,
            "Payload (octets)": (payload + 7) // 8,
            "Gain (%)": gain_pct(raw_bits, payload),
            "RC (×)": (raw_bits / payload) if payload else 0.0,
            "Encodage (ms)": enc_s * 1000,
            "Décodage (ms)": dec_s * 1000,
            "Total (ms)": total_s * 1000,
            "Latence (µs/pixel)": total_s * 1e6 / n_pixels,
            "Sans perte": "OK" if ok else "ÉCHEC",
            "Mesure": "réelle",
        })
        verified_labels.add(row["Pipeline"])
        return ok

    # ── 1. Balayage par niveaux, avec élagage au-delà de 2 étapes ───────────
    start = {
        "seq": list(symbols), "names": [], "plain_bits": raw_bits,
        "enc_s": 0.0, "dec_s": 0.0, "ok": True, "last": None,
    }
    start["score"] = add_variants(start)
    frontier = [start] if max_depth > 0 else []
    for depth in range(1, max_depth + 1):
        children = []
        for state in frontier:
            used = set(state["names"])
            for stage_name in stage_names:
                if stage_name in used:
                    continue
                if stage_name == "Delta" and state["names"]:
                    continue  # Delta s'applique aux pixels bruts : première étape
                child = expand(state, stage_name)
                if child is None:
                    continue
                child["score"] = add_variants(child)
                if child["ok"]:
                    children.append(child)
        if not children:
            break
        frontier = (select_frontier(children, beam_width)
                    if guided and depth >= 2 else children)

    # ── 2. Vérification réelle des meilleurs candidats ──────────────────────
    estimated = [r for r in results
                 if r["_coder"] and str(r["Sans perte"]).startswith("OK")]
    estimated.sort(key=lambda r: r["Payload (bits)"])
    picks, picked = [], set()

    def allowed(row):
        return not (row["_coder"] == "Arithmétique"
                    and len(symbols) > COMPARE_ARITH_VERIFY_MAX)

    # Un candidat vérifié par codeur terminal (le meilleur de chaque famille).
    for coder_name in coder_names:
        for row in estimated:
            if row["_coder"] == coder_name and allowed(row):
                if row["Pipeline"] not in picked:
                    picks.append(row)
                    picked.add(row["Pipeline"])
                break
    # Puis compléter avec les meilleures chaînes.
    for row in estimated:
        if len(picks) >= verify_top:
            break
        if allowed(row) and row["Pipeline"] not in picked:
            picks.append(row)
            picked.add(row["Pipeline"])
    picks = picks[:verify_top]

    # Vérifie aussi la meilleure chaîne « compacte » (Delta/RLE sans LZ/BWT) :
    # c'est elle qui donne le plus petit conteneur en mode tuiles, car les
    # grandes tables LZ/BWT seraient répétées dans chaque tuile.
    tile_tokens = {"BWT", "LZ77", "LZ78", "LZW"}
    compact = [r for r in estimated
               if not any(name in tile_tokens for name in r["_names"])
               and r["_coder"] != "Arithmétique"]
    chosen_labels = {r["Pipeline"] for r in picks}
    for row in compact:
        if row["Pipeline"] not in chosen_labels and allowed(row):
            picks.append(row)
            break

    for i, row in enumerate(picks):
        push_progress(0.88 + 0.12 * i / max(len(picks), 1))
        verify_row(row)
    push_progress(1.0)

    df = pd.DataFrame(results).drop(columns=["_names", "_coder"], errors="ignore")
    for col in ("Encodage (ms)", "Décodage (ms)", "Total (ms)", "Latence (µs/pixel)"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["Payload (bits)", "Total (ms)"],
                        na_position="last").reset_index(drop=True)
    df.index += 1
    info = {
        "n_symbols": len(symbols),
        "guided": bool(guided),
        "beam_width": int(beam_width),
        "max_depth": int(max_depth),
        "planned_rows": int(expected),
        "n_rows": len(df),
        "n_verified": len(verified_labels),
        "n_estimated": int((df["Mesure"] == "estimée").sum()),
    }
    return df, skipped, info


def render_compare(res):
    """Affiche le tableau comparatif."""
    df = res["df"]
    st.subheader("4. Comparatif des pipelines")

    n_sample = int(res["n_symbols"])
    n_total = int(res.get("n_total", n_sample))
    pct = (n_sample / n_total * 100) if n_total else 0.0
    pct_text = f"{pct:.2f} %" if pct < 1 else f"{pct:.1f} %"
    search_mode = (f"recherche guidée (faisceau {res.get('beam_width', '—')})"
                   if res.get("guided", True) else "recherche exhaustive")
    st.caption(
        f"Échantillon : **{n_sample:,} / {n_total:,} valeurs RGB** ({pct_text}), "
        f"réparti en {res.get('n_chunks', 1)} bloc(s) · {search_mode} · "
        f"{res.get('n_verified', 0)} chaîne(s) vérifiée(s) avec le vrai codeur, "
        f"{res.get('n_estimated', 0)} ligne(s) non vérifiée(s)."
    )

    hide_negative = st.checkbox("Masquer les gains négatifs (comme la console)", value=True, key="hide_neg")
    shown = df[df["Gain (%)"] >= 0] if hide_negative else df
    hidden = len(df) - len(shown)

    ok_mask = df["Sans perte"].astype(str).str.startswith("OK")
    candidates = df[(df["Catégorie"] != "référence") & ok_mask]

    def is_coded(label):
        """Vrai si la chaîne se termine par un codeur entropique."""
        label = str(label)
        return "Huffman" in label or "Arithmétique" in label

    coded = candidates[candidates["Pipeline"].map(is_coded)]
    best = coded.iloc[0] if len(coded) else None
    positive = coded[coded["Gain (%)"] > 0]
    measured = positive[positive["Total (ms)"].notna()]
    fastest = measured.sort_values("Total (ms)").iloc[0] if len(measured) else None

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pipelines évalués", f"{len(df):,}",
              help=f"{hidden} masqué(s) (gain < 0)" if hide_negative else None)
    if best is not None:
        k2.metric("Meilleur gain payload (codeur)", f"{best['Gain (%)']:.4f} %",
                  delta=best["Pipeline"], delta_color="off")
    else:
        k2.metric("Meilleur gain payload (codeur)", "—",
                  help="Sélectionne au moins un codeur terminal (Huffman ou Arithmétique).")
    if fastest is not None:
        k3.metric("Plus rapide (codeur, mesuré, gain > 0)", fmt_ms(fastest["Total (ms)"]),
                  delta=fastest["Pipeline"], delta_color="off")
    else:
        k3.metric("Plus rapide (codeur, mesuré, gain > 0)", "—",
                  help="Aucune chaîne à la fois gagnante et chronométrée. "
                       "Utilise le mode « Chaîne unique » pour un chronométrage complet.")
    k4.metric("Temps du comparatif", fmt_ms(res["elapsed_s"] * 1000))

    if not len(coded):
        st.warning("Aucun codeur terminal : les gains des chaînes « sans entropie » sont "
                   "théoriques. Sélectionne **Huffman** (ou Arithmétique) pour un vrai "
                   "fichier compressé.")
    else:
        verified = coded[coded["Mesure"] == "réelle"]
        if len(verified):
            vbest = verified.iloc[0]
            vrc = (n_sample * 8) / vbest["Payload (bits)"] if vbest["Payload (bits)"] else 0.0
            with st.container(border=True):
                st.markdown(
                    f"**MSE** = 0 · **SSIM** = 1.000000 · **RC** = {vrc:.2f} : 1 · "
                    f"**SNR** = ∞ dB · **PSNR** = ∞ dB"
                )
                st.caption(f"Meilleure chaîne vérifiée sans perte sur l'échantillon : {vbest['Pipeline']}")

    # Recommandation pour le test complet : une chaîne mesurée avec un codeur.
    # En mode tuiles, on privilégie les alphabets compacts (Delta/RLE) : les
    # tables LZ/BWT seraient répétées dans chaque tuile et peuvent dominer le
    # fichier final, même si le payload seul paraît minuscule.
    if len(coded) and best is not None and len(measured):
        tile_tokens = "BWT|LZ77|LZ78|LZW"
        huffman_measured = measured[
            ~measured["Pipeline"].str.contains("Arithmétique", regex=False)]
        compact = huffman_measured[
            ~huffman_measured["Pipeline"].str.contains(tile_tokens, regex=True)]
        if len(compact):
            recommended = compact.sort_values("Gain (%)", ascending=False).iloc[0]
            tile_ready = True
        elif len(huffman_measured):
            recommended = huffman_measured.sort_values("Total (ms)").iloc[0]
            tile_ready = False
        else:
            recommended = None
            tile_ready = False

        if recommended is None:
            st.warning(
                "Aucune chaîne **Huffman** mesurée : pour l'image entière par tuiles, "
                "ajoute Huffman dans les codeurs. L'Arithmétique est beaucoup plus "
                "lente sur des tuiles photo et n'apporte presque rien."
            )
        else:
            st.info(
                f"Pour le test complet sur l'image entière par tuiles, privilégie "
                f"**{recommended['Pipeline']}** : gain payload "
                f"{recommended['Gain (%)']:.2f} % (échantillon), temps mesuré "
                f"{fmt_ms(recommended['Total (ms)'])} sur l'échantillon."
            )
            if tile_ready:
                st.caption("Choix orienté tuiles : cette chaîne garde des tables compactes, "
                           "contrairement aux alphabets LZ/BWT qui seraient répétés dans "
                           "chaque tuile.")
            else:
                st.caption("Aucune chaîne compacte mesurée : les tables LZ/BWT sont répétées "
                           "dans chaque tuile et le conteneur final peut être gros. Vérifie "
                           "l'estimation de taille en mode « Chaîne unique ».")
            st.button(
                "⚡ Tester cette chaîne sur l'image entière (tuiles)",
                on_click=apply_pipeline,
                args=(recommended["Pipeline"],),
                type="primary",
                help="Bascule en mode « Chaîne unique », sélectionne cette chaîne et "
                     "active la compression par tuiles sans perte.",
            )
            if ("Arithmétique" in str(best["Pipeline"])
                    and "Arithmétique" not in str(recommended["Pipeline"])):
                st.caption("Le meilleur gain payload vient de l'Arithmétique, mais elle est "
                           "beaucoup plus lente en mode tuiles ; la chaîne ci-dessus est "
                           "préférable.")

    if df["Sans perte"].astype(str).eq("ÉCHEC").any():
        st.error("Au moins une chaîne n'a pas pu être restituée sans perte (voir colonne « Sans perte »).")

    stretch(st.dataframe, shown, column_config={
        "Gain (%)": st.column_config.NumberColumn(format="%.4f"),
        "RC (×)": st.column_config.NumberColumn(
            format="%.3f", help="Ratio de compression = bits bruts / payload."),
        "Encodage (ms)": st.column_config.NumberColumn(format="%.3f"),
        "Décodage (ms)": st.column_config.NumberColumn(format="%.3f"),
        "Total (ms)": st.column_config.NumberColumn(format="%.3f"),
        "Latence (µs/pixel)": st.column_config.NumberColumn(format="%.2f"),
        "Mesure": st.column_config.TextColumn(
            "Mesure", help="« réelle » : encodage + décodage effectués ; "
                           "« estimée » : payload calculé sans exécuter le décodage "
                           "du codeur (Huffman exact, Arithmétique estimé par entropie)."),
    })
    st.caption(
        f"Référence : {n_sample:,} octets RGB bruts ({n_sample * 8:,} bits), payload seul. "
        "**Huffman** est exact ; **Arithmétique** est estimée par l'entropie pour les lignes "
        "« estimée » (le vrai codeur, très coûteux, n'est exécuté que sur les meilleurs candidats). "
        "Encodage/Décodage cumulent toutes les étapes et le codeur terminal quand il est mesuré. "
        "Latence = (encodage + décodage) / nombre de pixels. "
        "Les lignes « sans entropie » donnent un débit théorique après préprocesseurs : le "
        "conteneur stocke alors le flux Python brut, plus gros. Pour le test complet, choisis "
        "une chaîne avec **Huffman** ou **Arithmétique**. "
        "Ce tableau ne compte que le payload : en mode tuiles, les tables sont répétées dans "
        "chaque tuile et le conteneur peut être plus gros. L'estimation de taille s'affiche "
        "en mode « Chaîne unique »."
    )

    top = shown.head(15)
    if len(top):
        st.markdown("**Top 15 — gain payload (%)**")
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


def apply_pipeline(pipeline):
    """Applique une chaîne recommandée en mode tuiles (image entière)."""
    st.session_state["mode"] = "Chaîne unique"
    st.session_state["chosen_algos"] = [name for name in pipeline.split("+") if name]
    st.session_state["use_tiles"] = True
    st.session_state["tile_size"] = 64


st.sidebar.subheader("Algorithmes")
mode = st.sidebar.radio("Mode", ["Chaîne unique", "Comparatif automatique"], key="mode")

if mode == "Chaîne unique":
    if "chosen_algos" not in st.session_state:
        st.session_state["chosen_algos"] = ["Delta", "Huffman"]
    chosen_algos = st.sidebar.multiselect(
        "Algorithmes (ordre d'application)",
        STAGE_NAMES + CODER_NAMES,
        key="chosen_algos",
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

    if "use_tiles" not in st.session_state:
        st.session_state["use_tiles"] = False
    use_tiles = st.sidebar.checkbox(
        "Compresser l'image entière par tuiles",
        key="use_tiles",
        help="Découpe l'image en blocs et compresse chacun indépendamment : permet de traiter "
             "une grande image SANS PERTE malgré les limites des algorithmes en Python pur. "
             "Les métadonnées sont répétées dans chaque tuile.",
    )
    tile_size = TILE_CHOICES[2]
    if use_tiles:
        if "tile_size" not in st.session_state:
            st.session_state["tile_size"] = 64
        tile_size = st.sidebar.select_slider(
            "Taille des tuiles (px)", options=TILE_CHOICES, key="tile_size")
        st.sidebar.caption(
            f"Une tuile de {tile_size}×{tile_size} contient {tile_size * tile_size * 3:,} valeurs "
            "(BWT : 64 000 max, Arithmétique : 30 000 max)."
        )
        if entropy_name == "Arithmétique":
            st.sidebar.warning(
                "Arithmétique + tuiles est très lent (souvent 20× plus qu'Huffman) "
                "pour un gain souvent inférieur à 1 %. Préfère Delta → Huffman."
            )
else:
    use_tiles, tile_size = False, 0
    compare_stages = st.sidebar.multiselect(
        "Préprocesseurs à combiner", STAGE_NAMES, default=["Delta", "RLE", "LZ77"])
    compare_depth = st.sidebar.slider("Longueur max des chaînes", 1, 4, 2)
    compare_coders = st.sidebar.multiselect(
        "Codeurs terminaux", CODER_NAMES, default=CODER_NAMES)
    compare_guided = st.sidebar.checkbox(
        "Recherche guidée (recommandée)", value=True,
        help="Évalue toutes les chaînes de 1–2 étapes, puis ne prolonge que les "
             "meilleures branches pour les étapes 3–4 : beaucoup plus rapide, avec "
             "un classement quasi identique.")
    compare_beam = COMPARE_DEFAULT_BEAM
    if compare_guided:
        compare_beam = st.sidebar.slider(
            "Largeur du faisceau", 6, 32, COMPARE_DEFAULT_BEAM, step=2,
            help="Nombre de branches conservées à chaque niveau. Plus grand = plus "
                 "exhaustif, mais plus lent.")
    compare_sample = st.sidebar.select_slider(
        "Échantillon du comparatif (max valeurs RGB)",
        options=COMPARE_SAMPLE_CHOICES, value=COMPARE_DEFAULT_SAMPLE,
        help="Le comparatif classe les pipelines sur un échantillon réparti dans "
             "l'image. La compression réelle (mode « Chaîne unique ») utilise toute "
             "l'image : le comparatif ne sert qu'à choisir la meilleure chaîne.")
    config_error = None
    if compare_stages:
        est = estimate_pipeline_rows(
            compare_stages, min(compare_depth, len(compare_stages)),
            len(compare_coders), guided=compare_guided, beam_width=compare_beam)
        st.sidebar.caption(
            f"≈ {est:,} pipelines à évaluer · échantillon ≤ {compare_sample:,} valeurs")
        if compare_guided:
            st.sidebar.caption(
                "Delta n'est appliqué qu'en premier. Toutes les chaînes de 1–2 étapes "
                "sont testées ; au-delà, seules les meilleures branches sont prolongées.")
        elif compare_sample > 20_000:
            st.sidebar.warning(
                "Mode exhaustif sur un grand échantillon : le calcul peut durer. "
                "Réduis l'échantillon ou active la recherche guidée.")

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
    if mode == "Chaîne unique" and not use_tiles and original_array.size > MAX_SYMBOLS:
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
        tile_word = "tuile" if n_tiles_est == 1 else "tuiles"
        st.write(f"Mode tuiles : **{n_tiles_est:,} {tile_word}** de {tile_size} × {tile_size} pixels "
                 "(image entière, sans perte).")
        if chain:
            raw_bytes_est = int(original_array.nbytes)
            candidates, errors, seen = [], [], set()

            def add_candidate(stages, coder, label, pipeline):
                key = (tuple(stages), coder)
                if key in seen:
                    return
                seen.add(key)
                # Arithmétique est déjà très lente : une seule tuile suffit.
                count = 1 if coder == "Arithmétique" else 3
                tiles = representative_tiles(original_array, tile_size, count)
                benches = [benchmark_tile(t.tobytes(), t.shape,
                                          tuple(stages), coder) for t in tiles]
                failure = next((b["error"] for b in benches if "error" in b), None)
                if failure:
                    errors.append((label, failure))
                    return
                n = len(benches)

                def avg(name):
                    return sum(b[name] for b in benches) / n

                payload_tile = (round(avg("payload_bits")) + 7) // 8
                blob_tile = avg("blob_size")
                meta_tile = max(blob_tile - payload_tile, 0)
                candidates.append({
                    "label": label,
                    "pipeline": pipeline,
                    "current": pipeline == "+".join(chain),
                    "per_tile_s": avg("per_tile_s"),
                    "gain": gain_pct(avg("raw_bits"), avg("payload_bits")),
                    "payload": payload_tile * n_tiles_est,
                    "container": blob_tile * n_tiles_est,
                    "meta_pct": (100.0 * meta_tile / blob_tile if blob_tile else 0.0),
                })

            add_candidate(selected_stages, entropy_name, " → ".join(chain),
                          "+".join(chain))
            add_candidate(("Delta",), "Huffman", "Delta → Huffman", "Delta+Huffman")
            add_candidate(("Delta", "RLE"), "Huffman",
                          "Delta → RLE → Huffman", "Delta+RLE+Huffman")

            if candidates:
                fast_candidates = [c for c in candidates
                                   if not c["pipeline"].endswith("Arithmétique")]
                best_container = min(fast_candidates or candidates,
                                     key=lambda c: c["container"])
                rows = []
                for c in sorted(candidates, key=lambda c: c["container"]):
                    status = []
                    if c["current"]:
                        status.append("actuelle")
                    if c is best_container:
                        status.append("recommandée")
                    rows.append({
                        "Chaîne": c["label"],
                        "Gain tuile (%)": f"{c['gain']:.1f}",
                        "Payload estimé": fmt_bytes(c["payload"]),
                        "Métadonnées": f"{c['meta_pct']:.0f} %",
                        "Conteneur estimé": fmt_bytes(c["container"]),
                        "Durée estimée": fmt_duration(c["per_tile_s"] * n_tiles_est),
                        "Statut": " · ".join(status) or "—",
                    })
                st.caption("Estimation d'après quelques tuiles réelles réparties dans "
                           f"l'image, pour {n_tiles_est:,} {tile_word}. Le conteneur inclut "
                           "les tables répétées dans chaque tuile.")
                stretch(st.dataframe, pd.DataFrame(rows), hide_index=True)

                current = next((c for c in candidates if c["current"]), None)
                if current is not None:
                    if current["container"] > raw_bytes_est:
                        st.error(
                            f"Conteneur estimé de la chaîne actuelle : "
                            f"**≈ {fmt_bytes(current['container'])}**, soit plus que les "
                            f"pixels bruts ({fmt_bytes(raw_bytes_est)}). Les métadonnées "
                            f"représentent {current['meta_pct']:.0f} % du fichier. "
                            "Préfère une chaîne compacte comme Delta → RLE → Huffman.")
                    elif current["meta_pct"] >= 50:
                        st.warning(
                            f"Les métadonnées représentent {current['meta_pct']:.0f} % du "
                            "conteneur : les tables par tuile dominent. Une chaîne sans "
                            "LZ/BWT produira un fichier plus petit.")
                    elif (best_container["pipeline"] != current["pipeline"]
                          and best_container["container"] < current["container"] * 0.85):
                        st.info(
                            f"Pour un conteneur plus petit, essaie "
                            f"**{best_container['label']}** : "
                            f"≈ {fmt_bytes(best_container['container'])} au lieu de "
                            f"≈ {fmt_bytes(current['container'])}.")

                    est_current_s = current["per_tile_s"] * n_tiles_est
                    if est_current_s > 300:
                        st.warning(
                            f"Durée estimée de la chaîne actuelle : "
                            f"**≈ {fmt_duration(est_current_s)}**. Pour aller plus vite, "
                            "choisis une chaîne Huffman compacte : Delta → Huffman ou "
                            "Delta → RLE → Huffman.")

                if best_container["pipeline"] != "+".join(chain):
                    st.button(
                        f"Appliquer {best_container['label']}",
                        on_click=apply_pipeline,
                        args=(best_container["pipeline"],),
                        help="Sélectionne cette chaîne dans la barre latérale.",
                    )
            for label, error in errors:
                st.caption(f"Estimation indisponible pour {label} : {error}")
else:
    chain = []
    signature = ("compare", chosen_name, selected_path.stat().st_mtime, max_side,
                 tuple(compare_stages), compare_depth, tuple(compare_coders),
                 compare_sample, compare_guided, compare_beam)
    st.write("Préprocesseurs :", ", ".join(compare_stages) or "—",
             "· codeurs :", ", ".join(compare_coders) or "aucun",
             f"· longueur max : {compare_depth}")
    n_total_values = int(original_array.size)
    n_sample_max = min(compare_sample, n_total_values)
    pct = (n_sample_max / n_total_values * 100) if n_total_values else 0.0
    pct_text = f"{pct:.2f} %" if pct < 1 else f"{pct:.1f} %"
    st.caption(
        f"Comparatif sur un échantillon de **≤ {n_sample_max:,} valeurs** "
        f"({pct_text} de l'image), réparti en blocs. "
        "La compression finale (sans perte) se fait sur toute l'image en mode « Chaîne unique »."
    )

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

            if max(reconstructed.size) > 1200:
                scale = 1200 / max(reconstructed.size)
                preview_size = (max(1, round(reconstructed.width * scale)),
                                max(1, round(reconstructed.height * scale)))
                preview_rebuilt = reconstructed.resize(preview_size, Image.LANCZOS)
            else:
                preview_rebuilt = reconstructed
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
            flat = original_array.reshape(-1)
            sample, n_chunks, n_total = sample_flat_array(flat, compare_sample)
            symbols = sample.tolist()
            bar = st.progress(0.0, text="Évaluation des pipelines…")
            started = time.perf_counter()
            df, skipped, info = run_comparison(
                symbols, compare_stages, min(compare_depth, len(compare_stages)),
                compare_coders, guided=compare_guided, beam_width=compare_beam,
                progress=lambda p: bar.progress(p, text="Évaluation des pipelines…"))
            elapsed = time.perf_counter() - started
            bar.empty()
            data = dict(info)
            data.update({
                "df": df, "skipped": skipped, "elapsed_s": elapsed,
                "n_total": n_total, "n_chunks": n_chunks,
                "stem": selected_path.stem,
            })
            st.session_state["result"] = {"signature": signature, "kind": "compare", "data": data}
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
