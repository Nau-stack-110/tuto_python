#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Onglet pédagogique de compression de textes, moteur Rust sans perte."""

import codecs
from pathlib import Path
import shutil
import subprocess
import time

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
TEXT_DIR = ROOT / "texts"
ORIGINAL_DIR = TEXT_DIR / "original"
COMPRESSED_DIR = TEXT_DIR / "compressed"
RUST_DIR = ROOT / "rust_codec"
RUST_BIN = RUST_DIR / "target" / "release" / "rust_codec"
ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
COMPRESSED_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_MAX_BYTES = 50_000_000

TEXT_EXTENSIONS = [
    "txt", "json", "jsonl", "csv", "tsv", "md", "xml", "yaml", "yml",
    "toml", "ini", "cfg", "conf", "log", "html", "htm", "css", "js", "ts",
    "py", "rs", "c", "h", "cpp", "hpp", "java", "sql", "tex", "svg",
]

PIPELINES = {
    "⚡ Auto — meilleure chaîne Rust": "auto",
    "🧪 Chaîne personnalisée…": "custom",
    "BWT + RLE + Huffman (texte)": "bwt+rle+huffman",
    "BWT + RLE + Arithmétique": "bwt+rle+arithmetic",
    "BWT + Huffman": "bwt+huffman",
    "LZ77 + Huffman (rapide)": "lz77+huffman",
    "LZ77 + Arithmétique": "lz77+arithmetic",
    "LZ78 + Huffman": "lz78+huffman",
    "LZ78 + Arithmétique": "lz78+arithmetic",
    "LZW + Huffman": "lzw+huffman",
    "LZW + Arithmétique": "lzw+arithmetic",
    "RLE + Huffman": "rle+huffman",
    "Huffman seul": "huffman",
    "Arithmétique seule": "arithmetic",
    "BWT + RLE": "bwt+rle",
    "LZ78 seul": "lz78",
    "LZW seul": "lzw",
    "LZ77 seul": "lz77",
    "RLE seul": "rle",
    "Delta + Huffman": "delta+huffman",
}

def fmt_bytes(n):
    n = float(n)
    for unit, div in (("Mo", 1_048_576), ("Ko", 1024)):
        if abs(n) >= div:
            return f"{n / div:,.2f} {unit}"
    return f"{int(n):,} octets"


def fmt_ms(ms):
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    return f"{ms:.2f} ms"


@st.cache_resource(show_spinner=False)
def rust_binary():
    """Compile le binaire Rust une seule fois, sans dépendance externe."""
    if RUST_BIN.is_file():
        return RUST_BIN
    if not RUST_DIR.is_dir():
        raise RuntimeError(
            f"Dossier Rust introuvable : {RUST_DIR}. "
            "Le moteur doit être placé à côté de l'application."
        )
    cargo = shutil.which("cargo")
    if cargo is None:
        raise RuntimeError(
            "Rust/cargo n'est pas installé. Installe Rust, puis lance : "
            "cd rust_codec && cargo build --release --offline"
        )
    with st.spinner("Compilation du moteur Rust (aucune dépendance à télécharger)…"):
        proc = subprocess.run(
            [cargo, "build", "--release", "--offline"],
            cwd=RUST_DIR, capture_output=True, text=True, timeout=300,
        )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "compilation Rust échouée")
    if not RUST_BIN.is_file():
        raise RuntimeError("Le binaire Rust n'a pas été produit.")
    return RUST_BIN


def run_codec(args, timeout=300):
    """Appelle le binaire Rust et transforme sa sortie clé=valeur en dictionnaire."""
    binary = rust_binary()
    proc = subprocess.run(
        [str(binary), *args], capture_output=True, text=True, timeout=timeout,
    )
    values = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    if proc.returncode != 0 or values.get("status") != "ok":
        message = values.get("message") or proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(message or "échec du moteur Rust")
    return values


def run_codec_compare(args, timeout=600):
    """Appelle `rust_codec compare` et récupère les lignes de résultat."""
    binary = rust_binary()
    proc = subprocess.run(
        [str(binary), *args], capture_output=True, text=True, timeout=timeout,
    )
    values = {}
    rows = []
    for line in proc.stdout.splitlines():
        if line.startswith("result="):
            parts = line[len("result="):].split("\t")
            if len(parts) < 5:
                continue
            rows.append({
                "pipeline": parts[0],
                "payload_bits": int(parts[1]),
                "encode_ms": float(parts[2]),
                "decode_ms": float(parts[3]),
                "lossless": parts[4] == "1",
            })
        elif "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    if proc.returncode != 0 or values.get("status") != "ok":
        message = values.get("message") or proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(message or "échec du comparatif Rust")
    return values, rows


def _int(stats, key):
    return int(stats.get(key, 0))


def _float(stats, key):
    return float(stats.get(key, 0.0))


def _files_equal(path_a, path_b, chunk_size=1 << 20):
    """Compare deux fichiers par blocs, sans les charger entièrement en RAM."""
    with open(path_a, "rb") as file_a, open(path_b, "rb") as file_b:
        while True:
            block_a = file_a.read(chunk_size)
            block_b = file_b.read(chunk_size)
            if block_a != block_b:
                return False
            if not block_a:
                return True


def _save_upload(uploaded):
    token = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("text_last_upload") != token:
        target = ORIGINAL_DIR / Path(uploaded.name).name
        target.write_bytes(uploaded.getvalue())
        st.session_state["text_last_upload"] = token
        st.sidebar.success(f"Fichier enregistré : {target.name}")
        st.rerun()


def _pick_file():
    files = sorted(
        p for p in ORIGINAL_DIR.iterdir()
        if p.is_file() and not p.name.startswith(".")
    )
    if not files:
        return None
    names = [p.name for p in files]
    selected = st.sidebar.selectbox("Fichier texte", names, key="text_file")
    return ORIGINAL_DIR / selected


def _pipeline_slug(spec):
    return spec.replace("+", "_").replace("/", "_")


_STAGE_DISPLAY = {
    "rle": "RLE",
    "bwt": "BWT",
    "lz77": "LZ77",
    "lz78": "LZ78",
    "lzw": "LZW",
    "delta": "Delta",
    "huffman": "Huffman",
    "arithmetic": "Arithmétique",
    "arith": "Arithmétique",
}


def _pretty_pipeline(spec):
    """Nom lisible d'un pipeline Rust, ex. lz77+bwt+arithmetic."""
    if spec == "brut":
        return "UTF-8 brut"
    tokens = [token.strip() for token in spec.split("+") if token.strip()]
    return "+".join(_STAGE_DISPLAY.get(token.lower(), token) for token in tokens)


def _pipeline_key(label):
    """Clé commune entre les libellés compress_lab et les noms Rust."""
    if "brut" in label.lower():
        return "brut"
    label = label.split(" (")[0].strip()
    label = label.replace("RLE classique", "RLE")
    tokens = [token.strip().lower() for token in label.split("+") if token.strip()]
    return "+".join(
        "arithmetic" if token in ("arithmetic", "arith", "arithmétique")
        else "huffman" if token == "huffman"
        else token
        for token in tokens
    )


def _fmt_ms(value):
    if value is None:
        return "—"
    if value < 1:
        return f"{value * 1000:.0f} µs"
    return f"{value:.2f} ms"


def _render_stats(path, stats, output_path, oversized):
    input_bytes = _int(stats, "input_bytes")
    output_bytes = _int(stats, "output_bytes")
    payload_bytes = _int(stats, "payload_bytes")
    overhead = _int(stats, "container_overhead")
    encode_ms = _float(stats, "encode_ms")
    decode_ms = _float(stats, "decode_ms")
    gain_file = (input_bytes - output_bytes) / input_bytes * 100 if input_bytes else 0.0
    gain_payload = (input_bytes - payload_bytes) / input_bytes * 100 if input_bytes else 0.0

    st.markdown(f"**Chaîne réellement utilisée :** `{stats.get('pipeline', '—')}`")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fichier original", f"{input_bytes:,} o", help="Taille exacte du fichier source.")
    c2.metric("Fichier .rlab", f"{output_bytes:,} o",
              delta=f"{gain_file:+.2f} %", help="Fichier complet, en-tête et tables comprises.")
    c3.metric("Payload compressé", f"{payload_bytes:,} o",
              delta=f"{gain_payload:+.2f} %", help="Données compressées finales, sans en-tête.")
    c4.metric("Métadonnées", f"{overhead:,} o",
              help="En-tête .rlab, nom du pipeline et tables Huffman.")

    t1, t2 = st.columns(2)
    t1.metric("Encodage Rust", fmt_ms(encode_ms))
    t2.metric("Décodage + vérification", fmt_ms(decode_ms))

    if oversized:
        st.warning(
            f"Le conteneur (**{output_bytes:,} o**) est plus gros que le fichier original "
            f"(**{input_bytes:,} o**) : il n'est pas conservé. C'est typique d'un fichier "
            "très petit ou déjà incompressible. Le payload seul fait "
            f"{payload_bytes:,} o, mais l'en-tête et les tables coûtent {overhead:,} o."
        )
    elif stats.get("exact") == "1":
        st.success(
            "✅ Sans perte : le moteur Rust a reconstruit **exactement** les mêmes octets "
            "(vérification intégrale pendant la compression)."
        )
    else:
        st.error("La vérification sans perte a échoué : ne pas utiliser ce fichier.")

    if not oversized and output_path.is_file():
        if output_bytes <= DOWNLOAD_MAX_BYTES:
            st.download_button(
                "⬇️ Télécharger le fichier compressé (.rlab)",
                data=output_path.read_bytes(),
                file_name=output_path.name,
                mime="application/octet-stream",
                key=f"text_dl_{output_path.name}",
            )
        else:
            st.caption("Fichier trop volumineux pour le téléchargement navigateur ; "
                       f"il est disponible ici : `{output_path.relative_to(ROOT)}`")


def _render_decompression(path, output_path):
    key = f"text_decompressed_{output_path.name}"
    if st.button("🔍 Décompresser et vérifier les octets", key=f"text_verify_{output_path.name}"):
        reconstructed = COMPRESSED_DIR / f"{path.stem}_reconstruit{path.suffix or '.bin'}"
        try:
            run_codec(["decompress", str(output_path), str(reconstructed)])
            same = _files_equal(path, reconstructed)
            st.session_state[key] = {
                "path": str(reconstructed),
                "same": same,
                "bytes": reconstructed.stat().st_size,
            }
        except Exception as exc:
            st.error(f"Décompression impossible : {exc}")
            st.session_state.pop(key, None)

    result = st.session_state.get(key)
    if result and Path(result["path"]).is_file():
        if result["same"]:
            st.success(
                f"✅ Deuxième vérification réussie : {result['bytes']:,} octets strictement "
                "identiques au fichier original."
            )
        else:
            st.error("❌ Le fichier décompressé diffère de l'original.")
        if result["bytes"] <= DOWNLOAD_MAX_BYTES:
            st.download_button(
                "⬇️ Télécharger le fichier reconstruit",
                data=Path(result["path"]).read_bytes(),
                file_name=Path(result["path"]).name,
                mime="text/plain",
                key=f"text_dl_dec_{output_path.name}",
            )


REFERENCE_MAX_BYTES = 12_000
REFERENCE_MAX_STAGES = 4


def _compare_source_text(path):
    """Texte lu comme compress_lab.py : UTF-8, fins de ligne finales retirées."""
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None
    return text.rstrip("\n")


def _build_compare_rows(path, max_stages):
    text = _compare_source_text(path)
    source_path = path
    temp_path = None
    if text is not None:
        temp_path = COMPRESSED_DIR / f"{path.stem}_comparatif_source.txt"
        temp_path.write_text(text, encoding="utf-8")
        source_path = temp_path

    try:
        stats, rust_rows = run_codec_compare(
            ["compare", str(source_path), str(max_stages)], timeout=900,
        )
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    reference = None
    reference_elapsed = None
    reference_error = None
    if text is not None and path.stat().st_size <= REFERENCE_MAX_BYTES:
        try:
            import compress_lab

            started = time.perf_counter()
            reference = compress_lab.run_all_pipelines(
                text, max_stages=min(max_stages, REFERENCE_MAX_STAGES),
            )
            reference_elapsed = time.perf_counter() - started
        except Exception as exc:  # pragma: no cover - dépend de l'environnement
            reference_error = str(exc)
    elif text is not None:
        reference_error = (
            f"référence compress_lab non calculée au-delà de "
            f"{REFERENCE_MAX_BYTES:,} octets (fichier de {path.stat().st_size:,} o)"
        )

    rust_by_key = {row["pipeline"]: row for row in rust_rows}
    input_bytes = _int(stats, "input_bytes")
    raw_bits = _int(stats, "raw_bits")
    rows = []
    gain_ge0 = _int(stats, "gain_ge0")
    hidden = _int(stats, "hidden")

    if reference is not None:
        reference_bits = len(text.encode("utf-8")) * 8
        gain_ge0 = sum(
            1 for result in reference if result["payload"] <= reference_bits
        )
        hidden = len(reference) - gain_ge0
        for result in reference:
            key = _pipeline_key(result["label"])
            actual = rust_by_key.get(key)
            payload_bits = result["payload"]
            model_gain = (
                (reference_bits - payload_bits) / reference_bits * 100
                if reference_bits else 0.0
            )
            real_payload = actual["payload_bits"] // 8 if actual else None
            real_gain = (
                (input_bytes - real_payload) / input_bytes * 100
                if actual is not None and input_bytes else None
            )
            rows.append({
                "key": key,
                "Pipeline": _pretty_pipeline(key),
                "Payload modèle (bits)": payload_bits,
                "Octets modèle": payload_bits / 8,
                "Gain modèle (%)": round(model_gain, 4),
                "Payload Rust (o)": real_payload,
                "Gain Rust (%)": round(real_gain, 2) if real_gain is not None else None,
                "Encodage (ms)": round(actual["encode_ms"], 3) if actual else None,
                "Décodage (ms)": round(actual["decode_ms"], 3) if actual else None,
                "Sans perte": bool(result["ok"] and actual and actual["lossless"]),
            })
        seen = {row["key"] for row in rows}
        for actual in rust_rows:
            if actual["pipeline"] in seen:
                continue
            payload = actual["payload_bits"] // 8
            real_gain = (input_bytes - payload) / input_bytes * 100 if input_bytes else 0.0
            rows.append({
                "key": actual["pipeline"],
                "Pipeline": _pretty_pipeline(actual["pipeline"]),
                "Payload modèle (bits)": None,
                "Octets modèle": None,
                "Gain modèle (%)": None,
                "Payload Rust (o)": payload,
                "Gain Rust (%)": round(real_gain, 2),
                "Encodage (ms)": round(actual["encode_ms"], 3),
                "Décodage (ms)": round(actual["decode_ms"], 3),
                "Sans perte": actual["lossless"],
            })
    else:
        for actual in rust_rows:
            payload = actual["payload_bits"] // 8
            real_gain = (input_bytes - payload) / input_bytes * 100 if input_bytes else 0.0
            rows.append({
                "key": actual["pipeline"],
                "Pipeline": _pretty_pipeline(actual["pipeline"]),
                "Payload modèle (bits)": None,
                "Octets modèle": None,
                "Gain modèle (%)": None,
                "Payload Rust (o)": payload,
                "Gain Rust (%)": round(real_gain, 2),
                "Encodage (ms)": round(actual["encode_ms"], 3),
                "Décodage (ms)": round(actual["decode_ms"], 3),
                "Sans perte": actual["lossless"],
            })

    return {
        "rows": rows,
        "has_reference": reference is not None,
        "reference_error": reference_error,
        "reference_elapsed": reference_elapsed,
        "input_bytes": input_bytes,
        "raw_bits": raw_bits,
        "evaluated": _int(stats, "evaluated"),
        "gain_ge0": gain_ge0,
        "hidden": hidden,
        "total_ms": _float(stats, "total_ms"),
        "max_stages": max_stages,
        "signature": (
            str(path), path.stat().st_mtime_ns, path.stat().st_size, max_stages,
        ),
    }


def _render_compare_results(stored, top_choice, sort_choice, include_negative):
    rows = stored["rows"]
    has_reference = stored["has_reference"]

    if not include_negative:
        if has_reference:
            rows = [r for r in rows if r["Gain modèle (%)"] is not None
                    and r["Gain modèle (%)"] >= 0]
        else:
            rows = [r for r in rows if r["Gain Rust (%)"] is not None
                    and r["Gain Rust (%)"] >= 0]
    if not rows:
        st.warning("Aucun pipeline ne passe le filtre de gain.")
        return

    def speed(row):
        if row["Encodage (ms)"] is None:
            return float("inf")
        return row["Encodage (ms)"] + row["Décodage (ms)"]

    if sort_choice == "Payload réel Rust":
        rows = sorted(rows, key=lambda r: (
            r["Payload Rust (o)"] is None, r["Payload Rust (o)"] or 0, speed(r)))
    elif sort_choice == "Rapidité (encodage + décodage)":
        rows = sorted(rows, key=lambda r: (
            not r["Sans perte"], speed(r),
            r["Payload Rust (o)"] if r["Payload Rust (o)"] is not None else 10**18))
    elif sort_choice == "Gain réel":
        rows = sorted(rows, key=lambda r: (
            -(r["Gain Rust (%)"] if r["Gain Rust (%)"] is not None else -10**9),
            speed(r)))
    elif sort_choice == "Gain modèle" and has_reference:
        rows = sorted(rows, key=lambda r: (
            -(r["Gain modèle (%)"] if r["Gain modèle (%)"] is not None else -10**9),
            speed(r)))
    elif has_reference:
        rows = sorted(rows, key=lambda r: (
            r["Payload modèle (bits)"] is None,
            r["Payload modèle (bits)"] or 0, speed(r)))
    else:
        rows = sorted(rows, key=lambda r: (
            r["Payload Rust (o)"] is None, r["Payload Rust (o)"] or 0, speed(r)))

    def records(source):
        return [
            {
                "#": rank,
                "Pipeline": row["Pipeline"],
                "Payload modèle (bits)": row["Payload modèle (bits)"],
                "Octets modèle": row["Octets modèle"],
                "Gain modèle (%)": row["Gain modèle (%)"],
                "Payload Rust (o)": row["Payload Rust (o)"],
                "Gain Rust (%)": row["Gain Rust (%)"],
                "Encodage (ms)": row["Encodage (ms)"],
                "Décodage (ms)": row["Décodage (ms)"],
                "Sans perte": row["Sans perte"],
            }
            for rank, row in enumerate(source, 1)
        ]

    limit = len(rows) if top_choice == "Tout" else int(top_choice)
    display = rows[:limit]
    frame = pd.DataFrame(records(display))
    full_frame = pd.DataFrame(records(rows))
    if not has_reference:
        drop = ["Payload modèle (bits)", "Octets modèle", "Gain modèle (%)"]
        frame = frame.drop(columns=drop)
        full_frame = full_frame.drop(columns=drop)

    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        height=min(720, 38 + 35 * len(frame)),
        column_config={
            "Octets modèle": st.column_config.NumberColumn(format="%.4f"),
            "Gain modèle (%)": st.column_config.NumberColumn(format="%.4f"),
            "Gain Rust (%)": st.column_config.NumberColumn(format="%.2f"),
            "Encodage (ms)": st.column_config.NumberColumn(format="%.3f"),
            "Décodage (ms)": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    model_best = None
    if has_reference:
        model_best = min(
            (r for r in rows if r["Payload modèle (bits)"] is not None),
            key=lambda r: r["Payload modèle (bits)"],
            default=None,
        )
        if model_best:
            st.caption(
                f"🏆 Meilleur pipeline au sens **compress_lab.py** : "
                f"**{model_best['Pipeline']}** — payload modèle "
                f"**{model_best['Payload modèle (bits)']:,} bits** "
                f"({model_best['Octets modèle']:.4f} o, gain "
                f"{model_best['Gain modèle (%)']:.4f} %), vérifié sans perte."
            )

    real_best = min(
        (r for r in rows if r["Payload Rust (o)"] is not None and r["Sans perte"]),
        key=lambda r: r["Payload Rust (o)"],
        default=None,
    )
    if real_best:
        st.caption(
            f"⚙️ Meilleur **payload réellement produit par Rust** : "
            f"**{real_best['Pipeline']}** — {real_best['Payload Rust (o)']:,} o "
            f"(gain {real_best['Gain Rust (%)']:.2f} % vs entrée), "
            f"encodage {_fmt_ms(real_best['Encodage (ms)'])}, "
            f"décodage {_fmt_ms(real_best['Décodage (ms)'])}."
        )
        threshold = max(0.0, real_best["Gain Rust (%)"] * 0.9)
        fast = min(
            (r for r in rows if r["Sans perte"] and r["Gain Rust (%)"] is not None
             and r["Gain Rust (%)"] >= threshold),
            key=speed,
            default=None,
        )
        if fast:
            st.caption(
                f"🚀 Meilleur compromis gain/rapidité : **{fast['Pipeline']}** — "
                f"gain {fast['Gain Rust (%)']:.2f} %, "
                f"{_fmt_ms(fast['Encodage (ms)'] + fast['Décodage (ms)'])} au total."
            )

    if has_reference:
        st.caption(
            "Le modèle `compress_lab.py` mesure l'entropie du payload seul, sans les "
            "tables ni les dictionnaires : ses gains sont plus élevés que le flux "
            "réellement écrit par Rust. Les colonnes « Rust » donnent le payload et "
            "les temps réels du moteur sans perte."
        )
    else:
        st.caption(
            "Le payload Rust est le flux réellement produit (hors en-têtes) ; "
            "les temps sont mesurés pendant l'énumération et chaque pipeline est "
            "décompressé puis comparé à l'entrée."
        )

    path = Path(stored["signature"][0])
    actions = st.columns(2)
    if model_best and actions[0].button(
        f"🎯 Utiliser {model_best['Pipeline']} (modèle)",
        key=f"text_use_model_{path.name}_{stored['max_stages']}",
        help="Sélectionne ce pipeline dans la chaîne personnalisée ci-dessus.",
    ):
        st.session_state[f"text_pending_pipeline_{path.name}"] = model_best["key"]
        st.rerun()
    if real_best and actions[1].button(
        f"⚙️ Utiliser {real_best['Pipeline']} (payload réel)",
        key=f"text_use_real_{path.name}_{stored['max_stages']}",
        help="Sélectionne le pipeline qui produit le plus petit payload Rust.",
    ):
        st.session_state[f"text_pending_pipeline_{path.name}"] = real_best["key"]
        st.rerun()

    st.download_button(
        "⬇️ Télécharger le classement filtré (CSV)",
        data=full_frame.to_csv(index=False).encode("utf-8"),
        file_name=f"{Path(stored['signature'][0]).stem}_comparatif.csv",
        mime="text/csv",
        key=f"text_compare_csv_{stored['signature'][0]}_{stored['max_stages']}",
    )


def _comparison_table(path):
    st.markdown("#### 📊 Comparatif exhaustif des combinaisons")
    st.caption(
        "Comme dans la console `compress_lab.py` : toutes les séquences ordonnées de "
        "**RLE, BWT, LZ77, LZ78, LZW** (sans répétition), suivies ou non de "
        "**Huffman / Arithmétique**. Chaque pipeline est décompressé et vérifié."
    )
    cols = st.columns([1, 1, 1.4, 1])
    max_stages = cols[0].selectbox(
        "Étages max", [2, 3, 4], index=2,
        key=f"text_compare_stages_{path.name}",
        help="4 étages = 618 pipelines, comme le comparatif complet de compress_lab.py.",
    )
    top_choice = cols[1].selectbox(
        "Afficher", [10, 25, 50, 100, "Tout"], index=1,
        key=f"text_compare_top_{path.name}",
    )
    sort_choice = cols[2].selectbox(
        "Tri",
        ["Payload modèle (compress_lab)", "Payload réel Rust",
         "Rapidité (encodage + décodage)", "Gain modèle", "Gain réel"],
        index=0,
        key=f"text_compare_sort_{path.name}",
    )
    include_negative = cols[3].checkbox(
        "Gains < 0", value=False,
        key=f"text_compare_negative_{path.name}",
        help="Comme compress_lab.py, les pipelines à gain négatif sont masqués par défaut.",
    )

    signature = (
        str(path), path.stat().st_mtime_ns, path.stat().st_size, max_stages,
    )
    if st.button(
        "🔬 Lancer le comparatif complet",
        type="primary",
        key=f"text_compare_{path.name}_{max_stages}",
        help="Énumère toutes les combinaisons, vérifie chaque pipeline et mesure "
             "les temps réels du moteur Rust.",
    ):
        try:
            with st.spinner(
                "Énumération des combinaisons et vérification sans perte… "
                "(la référence compress_lab.py peut prendre quelques secondes)"
            ):
                stored = _build_compare_rows(path, max_stages)
            st.session_state[f"text_compare_result_{path.name}"] = stored
        except Exception as exc:
            st.error(f"Comparatif impossible : {exc}")
            st.session_state.pop(f"text_compare_result_{path.name}", None)

    stored = st.session_state.get(f"text_compare_result_{path.name}")
    if stored is None:
        st.info(
            "Le comparatif affiche les meilleures combinaisons triées par payload, "
            "avec le gain et les temps d'encodage/décodage réels."
        )
        return
    if stored["signature"] != signature:
        st.info("Les paramètres ont changé : relance le comparatif complet.")
        return

    metrics = st.columns(4)
    metrics[0].metric("Pipelines évalués", f"{stored['evaluated']:,}")
    metrics[1].metric(
        "Gain ≥ 0",
        f"{stored['gain_ge0']:,}" if stored["has_reference"] else "—",
        help="Nombre de pipelines dont le payload modèle ne dépasse pas l'entrée.",
    )
    metrics[2].metric(
        "Masqués",
        f"{stored['hidden']:,}" if stored["has_reference"] else "—",
        help="Pipelines à gain négatif, masqués par défaut.",
    )
    metrics[3].metric(
        "Temps Rust",
        _fmt_ms(stored["total_ms"]),
        help="Temps d'énumération, de vérification et de mesure des 618 pipelines.",
    )
    if stored["has_reference"]:
        st.caption(
            f"Référence `compress_lab.py` calculée en "
            f"{stored['reference_elapsed']:.2f} s · "
            f"{stored['input_bytes']:,} octets comparés "
            "(fin de ligne finale retirée, comme dans compress_lab.py)."
        )
    elif stored["reference_error"]:
        st.caption(stored["reference_error"])

    _render_compare_results(stored, top_choice, sort_choice, include_negative)


def render_text_lab():
    st.subheader("📄 Compression de textes — sans perte")
    st.caption(
        "Moteur **Rust** (Delta, RLE, BWT, LZ77, LZ78, LZW, Huffman et arithmétique) · "
        "vérification intégrale des octets · aucune dépendance Rust externe."
    )

    st.sidebar.header("📄 Fichier texte")
    uploaded = st.sidebar.file_uploader(
        "Importer un fichier texte",
        type=TEXT_EXTENSIONS,
        key="text_upload",
    )
    if uploaded is not None:
        _save_upload(uploaded)

    path = _pick_file()

    with st.sidebar.expander("Ou créer un fichier depuis un texte", expanded=False):
        pasted = st.text_area("Texte / JSON à tester", height=160, key="text_paste")
        if st.button("Utiliser ce texte", key="text_paste_button"):
            if pasted.strip():
                target = ORIGINAL_DIR / "collage.txt"
                target.write_text(pasted, encoding="utf-8")
                st.session_state["text_last_upload"] = None
                st.rerun()
            else:
                st.warning("Le texte est vide.")

    if path is None:
        st.info(
            "Importe un fichier `.txt`, `.json`, `.csv`, `.md`… ou colle un texte dans "
            "la barre latérale. Les fichiers sont stockés dans `texts/original/`."
        )
        st.code("mkdir -p texts/original texts/compressed")
        return

    size = path.stat().st_size
    with path.open("rb") as file:
        preview_bytes = file.read(65536)
    try:
        preview = codecs.getincrementaldecoder("utf-8")("strict").decode(
            preview_bytes, final=False)
        is_utf8 = True
    except UnicodeDecodeError:
        preview = preview_bytes.decode("utf-8", errors="replace")
        is_utf8 = False

    st.markdown(f"### `{path.name}`")
    info1, info2, info3 = st.columns(3)
    info1.metric("Taille", fmt_bytes(size))
    info2.metric("Octets", f"{size:,}")
    info3.metric("UTF-8 (aperçu)", "oui" if is_utf8 else "non (mode octets)")
    if not is_utf8:
        st.caption("Le fichier n'est pas de l'UTF-8 valide : la compression reste "
                   "strictement sans perte au niveau des octets.")

    with st.expander("Aperçu du contenu", expanded=False):
        st.code(preview[:4000] + ("\n…" if len(preview) > 4000 else ""),
                language="json" if path.suffix.lower() == ".json" else None)

    pending = st.session_state.pop(f"text_pending_pipeline_{path.name}", None)
    if pending:
        st.session_state[f"text_pipeline_{path.name}"] = "🧪 Chaîne personnalisée…"
        st.session_state[f"text_custom_{path.name}"] = pending

    st.markdown("#### Chaîne de compression")
    label = st.selectbox(
        "Algorithme Rust",
        list(PIPELINES),
        index=0,
        key=f"text_pipeline_{path.name}",
        help="Auto essaie plusieurs chaînes et garde le plus petit fichier .rlab.",
    )
    spec = PIPELINES[label]
    if spec == "custom":
        spec = st.text_input(
            "Pipeline Rust personnalisé",
            value="bwt+rle+huffman",
            key=f"text_custom_{path.name}",
            help="Étages séparés par « + ». Huffman ou arithmetic doit être en dernier. "
                 "Exemples : bwt+rle+huffman, lz78+arithmetic, lz77+huffman, bwt+rle.",
        ).strip()
        label = f"Personnalisée : {spec or '—'}"
        if not spec:
            st.warning("Saisis une chaîne, par exemple `bwt+rle+huffman`.")
    if spec == "auto":
        st.caption(
            "Auto teste les chaînes rapides, puis BWT (≤ 2 Mo), LZ78/LZW (≤ 4 Mo) "
            "et l'arithmétique (≤ 32 Mo). Pour un gros fichier, choisis explicitement "
            "BWT + RLE + Huffman si tu acceptes un peu plus de temps."
        )
    signature = (str(path), path.stat().st_mtime_ns, path.stat().st_size, spec)

    if st.button("🚀 Compresser avec Rust", type="primary", key=f"text_run_{path.name}"):
        output_path = COMPRESSED_DIR / f"{path.stem}_{_pipeline_slug(spec)}.rlab"
        try:
            stats = run_codec(["compress", str(path), str(output_path), spec])
            input_bytes = _int(stats, "input_bytes")
            output_bytes = _int(stats, "output_bytes")
            oversized = output_bytes >= input_bytes
            if oversized:
                output_path.unlink(missing_ok=True)
            st.session_state["text_result"] = {
                "signature": signature,
                "stats": stats,
                "output_path": str(output_path),
                "oversized": oversized,
                "path": str(path),
            }
            st.session_state.pop(f"text_decompressed_{output_path.name}", None)
        except Exception as exc:
            st.error(f"Compression Rust impossible : {exc}")
            st.session_state.pop("text_result", None)

    stored = st.session_state.get("text_result")
    if stored is not None:
        if stored["signature"] == signature:
            output_path = Path(stored["output_path"])
            _render_stats(path, stored["stats"], output_path, stored["oversized"])
            if not stored["oversized"] and output_path.is_file():
                _render_decompression(path, output_path)
        else:
            st.info("La configuration a changé depuis le dernier test : relance la compression.")

    st.divider()
    _comparison_table(path)

    st.divider()
    st.caption(
        "Le **payload** est la donnée compressée finale, sans en-tête, comme dans "
        "`compress_lab.py`. Le fichier `.rlab` ajoute le pipeline, les tables et une "
        "empreinte de vérification. Aucun fichier plus gros que l'original n'est conservé. "
        "BWT et LZ78/LZW sont plus lents que LZ77/Huffman sur les gros fichiers ; "
        "l'arithmétique est un codeur terminal, comme Huffman. Sur du texte, "
        "LZ77+Huffman ou BWT+RLE+Huffman donnent généralement les meilleurs résultats ; "
        "LZ78/LZW restent surtout pédagogiques."
    )
