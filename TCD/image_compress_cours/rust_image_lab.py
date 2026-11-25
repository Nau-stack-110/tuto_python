#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compression d'images RGB par tuiles avec le moteur Rust sans perte.

Chaque tuile choisit la plus petite chaîne parmi Delta/RLE/LZ77 + Huffman.
Le conteneur `.rtile` est vérifié intégralement (décompression + empreinte)
avant d'être écrit.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import text_lab

ROOT = Path(__file__).resolve().parent
COMPRESSED_DIR = ROOT / "images" / "compressed"
COMPRESSED_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_MAX_BYTES = 50_000_000

TILE_CHOICES = [32, 64, 128, 256]

PIPELINES = {
    "⚡ Auto — meilleure chaîne par tuile": "auto",
    "🧪 Chaîne personnalisée…": "custom",
    "Delta + RLE + Huffman": "delta+rle+huffman",
    "Delta + Huffman": "delta+huffman",
    "LZ77 + Huffman": "lz77+huffman",
    "RLE + Huffman": "rle+huffman",
    "Huffman seul": "huffman",
    "BWT + RLE + Huffman": "bwt+rle+huffman",
    "LZ78 + Huffman": "lz78+huffman",
    "LZW + Huffman": "lzw+huffman",
    "Arithmétique seule": "arithmetic",
}


def _slug(spec):
    return spec.replace("+", "_").replace("/", "_")


def _files_equal(path_a, path_b, chunk_size=1 << 20):
    with open(path_a, "rb") as file_a, open(path_b, "rb") as file_b:
        while True:
            block_a = file_a.read(chunk_size)
            block_b = file_b.read(chunk_size)
            if block_a != block_b:
                return False
            if not block_a:
                return True


def _preview_from_raw(path, width, height, max_side=1200):
    """Construit un aperçu sans charger toute l'image RGB en mémoire."""
    step = max(1, int(np.ceil(max(width, height) / max_side)))
    preview_h = (height + step - 1) // step
    preview_w = (width + step - 1) // step
    preview = np.empty((preview_h, preview_w, 3), dtype=np.uint8)
    with open(path, "rb") as file:
        for out_y in range(preview_h):
            y = min(out_y * step, height - 1)
            file.seek(y * width * 3)
            row = np.fromfile(file, dtype=np.uint8, count=width * 3)
            if row.size != width * 3:
                raise ValueError("aperçu impossible : fichier RGB tronqué")
            preview[out_y] = row.reshape(width, 3)[::step][:preview_w]
    return preview


def _parse_counts(stats):
    rows = []
    for item in stats.get("pipeline_counts", "").split(","):
        item = item.strip()
        if not item:
            continue
        name, _, count = item.rpartition(":")
        if name:
            rows.append({"Chaîne gagnante": name, "Tuiles": int(count)})
    return pd.DataFrame(rows)


def render_sidebar_controls():
    """Contrôles de la barre latérale, à appeler avant le rendu principal."""
    st.sidebar.header("🧩 Rust — tuiles")
    st.sidebar.caption(
        "Chaque tuile est compressée indépendamment, puis la plus petite chaîne "
        "est choisie. La vérification est faite tuile par tuile et sur l'image entière."
    )
    tile_size = st.sidebar.select_slider(
        "Taille des tuiles (px)",
        options=TILE_CHOICES,
        value=128,
        key="rust_tile_size",
        help="128 est un bon compromis vitesse / taux de compression. "
             "Les tuiles plus petites laissent plus de liberté au choix adaptatif.",
    )
    label = st.sidebar.selectbox(
        "Chaîne Rust",
        list(PIPELINES),
        index=0,
        key="rust_tile_pipeline",
        help="Auto teste Delta/RLE/LZ77 + Huffman sur chaque tuile. "
             "Les autres chaînes sont imposées à toutes les tuiles.",
    )
    spec = PIPELINES[label]
    if spec == "custom":
        spec = st.sidebar.text_input(
            "Pipeline personnalisé",
            value="delta+rle+huffman",
            key="rust_tile_custom_pipeline",
            help="Étages séparés par « + ». Huffman ou arithmetic doit être en dernier. "
                 "Exemples : delta+rle+huffman, lz77+huffman, bwt+rle+huffman.",
        ).strip()
        label = f"Personnalisée : {spec or '—'}"
        if not spec:
            st.sidebar.warning("Saisis une chaîne, par exemple `delta+rle+huffman`.")
    return tile_size, spec, label


def render_rust_tiles(
    pixel_array,
    path,
    original_file_size,
    resized,
    preview_array,
    tile_size,
    pipeline_spec,
    pipeline_label,
):
    st.subheader("3. Compression Rust par tuiles — sans perte")
    st.caption(
        "Le moteur Rust compresse les pixels RGB bruts, tuile par tuile. "
        "Le fichier `.rtile` contient le pipeline gagnant et les données compressées ; "
        "il n'est pas destiné à être ouvert comme une image."
    )

    if resized:
        st.warning(
            "Ce mode compresse l'image **entière** et compare le résultat au fichier "
            "source. Choisis « Original » dans « Taille de l'image testée » pour éviter "
            "une comparaison trompeuse."
        )
        return

    if not pipeline_spec:
        st.warning("Saisis une chaîne Rust, par exemple `delta+rle+huffman`.")
        return

    height, width, channels = pixel_array.shape
    if channels != 3:
        st.error("Le moteur Rust attend une image RGB (3 canaux).")
        return

    n_tiles = ((height + tile_size - 1) // tile_size) * (
        (width + tile_size - 1) // tile_size
    )
    st.write(
        f"Image : **{width} × {height}** · **{n_tiles:,} tuiles** de "
        f"{tile_size} × {tile_size} px · chaîne : **{pipeline_label}**"
    )

    signature = (
        str(path),
        path.stat().st_mtime_ns,
        path.stat().st_size,
        pixel_array.shape,
        tile_size,
        pipeline_spec,
    )
    run_clicked = st.button(
        "🚀 Compresser l'image avec Rust",
        type="primary",
        key="rust_tile_run",
        help="Écrit les pixels RGB dans un fichier temporaire, compresse par tuiles, "
             "puis vérifie que l'image reconstruite est strictement identique.",
    )

    if run_clicked:
        raw_path = COMPRESSED_DIR / f"{path.stem}_rust_source.raw"
        output_path = (
            COMPRESSED_DIR
            / f"{path.stem}_rust_tuiles{tile_size}_{_slug(pipeline_spec)}.rtile"
        )
        try:
            output_path.unlink(missing_ok=True)
            with raw_path.open("wb") as file:
                pixel_array.tofile(file)
            stats = text_lab.run_codec(
                [
                    "tile",
                    str(raw_path),
                    str(width),
                    str(height),
                    str(tile_size),
                    str(output_path),
                    pipeline_spec,
                ],
                timeout=3600,
            )
            input_bytes = int(stats.get("input_bytes", 0))
            output_bytes = int(stats.get("output_bytes", 0))
            oversized = output_bytes >= original_file_size
            if oversized:
                output_path.unlink(missing_ok=True)
            st.session_state["rust_tile_result"] = {
                "signature": signature,
                "stats": stats,
                "output_path": str(output_path),
                "oversized": oversized,
                "input_bytes": input_bytes,
                "original_file_size": original_file_size,
                "path": str(path),
                "tile_size": tile_size,
                "pipeline_label": pipeline_label,
                "width": width,
                "height": height,
            }
            st.session_state.pop("rust_tile_verify", None)
        except Exception as exc:
            st.error(f"Compression Rust impossible : {exc}")
            st.session_state.pop("rust_tile_result", None)
        finally:
            raw_path.unlink(missing_ok=True)

    stored = st.session_state.get("rust_tile_result")
    if stored is None:
        st.info(
            "Le résultat n'est conservé que s'il est plus petit que le fichier original. "
            "Sur un JPEG déjà compressé, le `.rtile` peut être plus gros : il sera alors "
            "supprimé et expliqué."
        )
        return
    if stored["signature"] != signature:
        st.info("La configuration a changé depuis le dernier test : relance la compression.")
        return

    _render_result(stored, preview_array, pixel_array)


def _render_result(stored, preview_array, pixel_array):
    stats = stored["stats"]
    output_path = Path(stored["output_path"])
    input_bytes = stored["input_bytes"]
    original_file_size = stored["original_file_size"]
    output_bytes = int(stats.get("output_bytes", 0))
    payload_bytes = int(stats.get("payload_bytes", 0))
    overhead = int(stats.get("container_overhead", 0))
    tiles = int(stats.get("tiles", 0))
    encode_ms = float(stats.get("encode_ms", 0.0))
    decode_ms = float(stats.get("decode_ms", 0.0))

    gain_file = (
        (original_file_size - output_bytes) / original_file_size * 100
        if original_file_size
        else 0.0
    )
    gain_raw = (input_bytes - payload_bytes) / input_bytes * 100 if input_bytes else 0.0

    st.markdown(f"**Chaîne :** `{stored['pipeline_label']}` · **{tiles:,} tuiles**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fichier original", text_lab.fmt_bytes(original_file_size))
    c2.metric(
        "Fichier .rtile",
        text_lab.fmt_bytes(output_bytes),
        delta=f"{gain_file:+.2f} %" if not stored["oversized"] else "supprimé",
        help="Taille du conteneur complet, en-têtes et métadonnées de tuiles comprises.",
    )
    c3.metric(
        "Payload compressé",
        text_lab.fmt_bytes(payload_bytes),
        delta=f"{gain_raw:+.2f} % vs RGB",
        help="Somme des données compressées des tuiles, sans métadonnées.",
    )
    c4.metric(
        "Métadonnées",
        text_lab.fmt_bytes(overhead),
        help="En-tête .rtile et pipeline/métadonnées de chaque tuile.",
    )

    t1, t2, t3 = st.columns(3)
    t1.metric("Pixels RGB bruts", text_lab.fmt_bytes(input_bytes))
    t2.metric("Encodage Rust", text_lab.fmt_ms(encode_ms))
    t3.metric("Décodage + vérification", text_lab.fmt_ms(decode_ms))

    if stored["oversized"]:
        st.warning(
            f"Le conteneur (**{output_bytes:,} o**) est plus gros que le fichier original "
            f"(**{original_file_size:,} o**) : il n'est pas conservé. C'est typique d'un "
            "JPEG déjà compressé, où repartir des pixels RGB coûte plus cher que le JPEG. "
            f"Le payload seul fait {payload_bytes:,} o, soit {gain_raw:.2f} % de gain "
            "sur les pixels bruts, mais l'en-tête et les métadonnées de tuiles ajoutent "
            f"{overhead:,} o."
        )
    else:
        st.success(
            "✅ Sans perte : le moteur Rust a décompressé le conteneur et retrouvé "
            "**exactement** les mêmes pixels RGB (vérification globale par empreinte)."
        )
        if output_path.is_file():
            if output_bytes <= DOWNLOAD_MAX_BYTES:
                st.download_button(
                    "⬇️ Télécharger le fichier compressé (.rtile)",
                    data=output_path.read_bytes(),
                    file_name=output_path.name,
                    mime="application/octet-stream",
                    key=f"rust_tile_dl_{output_path.name}",
                )
            else:
                st.caption(
                    "Fichier trop volumineux pour le téléchargement navigateur ; "
                    f"il est disponible ici : `{output_path.relative_to(ROOT)}`"
                )

    counts = _parse_counts(stats)
    if len(counts):
        st.markdown("#### 🧩 Répartition des chaînes gagnantes")
        st.dataframe(counts, width="stretch", hide_index=True)

    if not stored["oversized"] and output_path.is_file():
        _render_verification(stored, preview_array, pixel_array)


def _render_verification(stored, preview_array, pixel_array):
    output_path = Path(stored["output_path"])
    path = Path(stored["path"])
    width = stored["width"]
    height = stored["height"]
    key = f"rust_tile_verify_{output_path.name}"

    if st.button(
        "🔍 Décompresser et vérifier les pixels",
        key=f"rust_tile_verify_btn_{output_path.name}",
    ):
        raw_path = COMPRESSED_DIR / f"{path.stem}_rust_source.raw"
        reconstructed = COMPRESSED_DIR / f"{path.stem}_rust_reconstruit.raw"
        try:
            with raw_path.open("wb") as file:
                pixel_array.tofile(file)
            text_lab.run_codec(
                ["tile-decompress", str(output_path), str(reconstructed)],
                timeout=3600,
            )
            same = _files_equal(raw_path, reconstructed)
            preview = _preview_from_raw(reconstructed, width, height)
            st.session_state[key] = {
                "same": same,
                "bytes": reconstructed.stat().st_size,
                "preview": preview,
            }
        except Exception as exc:
            st.error(f"Vérification impossible : {exc}")
            st.session_state.pop(key, None)
        finally:
            raw_path.unlink(missing_ok=True)
            reconstructed.unlink(missing_ok=True)

    result = st.session_state.get(key)
    if not result:
        return
    if result["same"]:
        st.success(
            f"✅ Deuxième vérification réussie : {result['bytes']:,} octets RGB "
            "strictement identiques à l'image source."
        )
    else:
        st.error("❌ Les pixels reconstruits diffèrent de l'image source.")

    left, right = st.columns(2)
    with left:
        st.image(preview_array, caption="Image originale (aperçu)")
    with right:
        st.image(result["preview"], caption="Image reconstruite par Rust (aperçu)")
