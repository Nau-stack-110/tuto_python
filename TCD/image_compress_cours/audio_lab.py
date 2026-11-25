#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Onglet audio du laboratoire de compression.

Objectifs :
  • WAV/AIFF PCM entier  -> FLAC sans perte, vérifié par hash PCM ;
  • FLAC                 -> déjà sans perte, archivage exact .rlab possible ;
  • WAV flottant         -> archivage exact .rlab (FLAC ne code que l'entier) ;
  • MP3/OGG Vorbis/Opus  -> formats avec perte : archivage exact .rlab seulement,
                            avec un avertissement explicite.

Aucune limite de taille n'est imposée : les encodages et les vérifications
utilisent des outils en flux (flac, ffmpeg), et l'interface ne charge jamais
tout le PCM en mémoire.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

import text_lab

ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "audios"
ORIGINAL_DIR = AUDIO_DIR / "original"
COMPRESSED_DIR = AUDIO_DIR / "compressed"
PREVIEW_DIR = AUDIO_DIR / "previews"

for directory in (ORIGINAL_DIR, COMPRESSED_DIR, PREVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)

AUDIO_EXTENSIONS = ["wav", "flac", "aif", "aiff", "mp3", "ogg", "oga", "opus"]
FLAC_LEVELS = [0, 5, 8]
FLAC_LEVEL_LABELS = {
    0: "0 — très rapide",
    5: "5 — équilibré (recommandé)",
    8: "8 — compression maximale",
}
PREVIEW_SECONDS = 30
PREVIEW_RATE = 44_100
WAVEFORM_RATE = 8_000
WAVEFORM_POINTS = 2_000

# Techniques du lab appliquées aux octets du fichier audio.
LAB_SAMPLE_BYTES = 512_000
LAB_BWT_MAX_BYTES = 8_000_000
LAB_LZ_MAX_BYTES = 32_000_000
LAB_PIPELINES = {
    "⚡ Auto — meilleure chaîne Rust": "auto",
    "Delta + Huffman": "delta+huffman",
    "RLE + Huffman": "rle+huffman",
    "BWT + Huffman": "bwt+huffman",
    "BWT + RLE + Huffman": "bwt+rle+huffman",
    "LZ77 + Huffman": "lz77+huffman",
    "LZ78 + Huffman": "lz78+huffman",
    "LZW + Huffman": "lzw+huffman",
    "Huffman seul": "huffman",
    "Arithmétique seule": "arithmetic",
    "Delta seul": "delta",
    "RLE seul": "rle",
    "BWT seul": "bwt",
    "LZ77 seul": "lz77",
    "LZ78 seul": "lz78",
    "LZW seul": "lzw",
    "LZ77 + Arithmétique": "lz77+arithmetic",
    "BWT + RLE + Arithmétique": "bwt+rle+arithmetic",
    "🧪 Chaîne personnalisée…": "custom",
}
LAB_TEST_PIPELINES = [
    "delta",
    "rle",
    "bwt",
    "lz77",
    "lz78",
    "lzw",
    "huffman",
    "arithmetic",
    "delta+huffman",
    "rle+huffman",
    "bwt+huffman",
    "bwt+rle+huffman",
    "lz77+huffman",
    "lz78+huffman",
    "lzw+huffman",
    "auto",
]

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"
FLAC = shutil.which("flac") or "flac"

LOSSY_CODECS = {
    "mp3", "mp3float", "mp2", "vorbis", "opus", "aac", "ac3", "eac3",
    "dts", "wmav1", "wmav2", "wmapro", "amr_nb", "amr_wb", "speex",
    "cook", "atrac3", "atrac3p", "pcm_alaw", "pcm_mulaw",
}
FLOAT_SAMPLE_FORMATS = {"flt", "fltp", "dbl", "dblp"}
CODEC_LABELS = {
    "mp3": "MP3",
    "mp3float": "MP3",
    "vorbis": "Ogg Vorbis",
    "opus": "Opus",
    "flac": "FLAC",
    "aac": "AAC",
    "pcm_s16le": "PCM 16 bits",
    "pcm_s16be": "PCM 16 bits",
    "pcm_s24le": "PCM 24 bits",
    "pcm_s24be": "PCM 24 bits",
    "pcm_s32le": "PCM 32 bits",
    "pcm_s32be": "PCM 32 bits",
    "pcm_u8": "PCM 8 bits",
    "pcm_s8": "PCM 8 bits",
    "pcm_f32le": "PCM flottant 32 bits",
    "pcm_f32be": "PCM flottant 32 bits",
    "pcm_f64le": "PCM flottant 64 bits",
    "pcm_f64be": "PCM flottant 64 bits",
}


# ──────────────────────────────────────────────────────────────────────────────
# Utilitaires généraux
# ──────────────────────────────────────────────────────────────────────────────
def fmt_bytes(n: int | float) -> str:
    n = float(n)
    for unit, div in (("Mo", 1_048_576), ("Ko", 1024)):
        if abs(n) >= div:
            return f"{n / div:,.2f} {unit}"
    return f"{int(n):,} octets"


def fmt_duration(seconds: float) -> str:
    if not seconds or seconds < 0:
        return "—"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)} min {sec:04.1f} s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours} h {minutes:02d} min {sec:04.1f} s"


def fmt_ms(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    if ms < 1:
        return f"{ms * 1000:.0f} µs"
    return f"{ms:.2f} ms"


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _run(args: list[str], timeout: int = 3600, check: bool = True) -> subprocess.CompletedProcess:
    """Lance un outil externe en capturant proprement les erreurs."""
    try:
        proc = subprocess.run(
            [str(arg) for arg in args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Outil introuvable : {args[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Délai dépassé ({timeout} s) pour {args[0]}"
        ) from exc
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(detail or f"Échec de {args[0]} (code {proc.returncode})")
    return proc


def _files_equal(path_a: Path, path_b: Path, chunk_size: int = 1 << 20) -> bool:
    """Compare deux fichiers par blocs, sans les charger en mémoire."""
    with path_a.open("rb") as file_a, path_b.open("rb") as file_b:
        while True:
            block_a = file_a.read(chunk_size)
            block_b = file_b.read(chunk_size)
            if block_a != block_b:
                return False
            if not block_a:
                return True


# ──────────────────────────────────────────────────────────────────────────────
# Métadonnées et classification
# ──────────────────────────────────────────────────────────────────────────────
def probe_audio(path: Path) -> dict:
    """Interroge ffprobe et retourne les métadonnées utiles du premier flux audio."""
    proc = _run(
        [
            FFPROBE, "-v", "error", "-show_format", "-show_streams",
            "-of", "json", str(path),
        ],
        timeout=180,
    )
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe n'a pas produit de JSON valide : {exc}") from exc

    streams = [
        stream for stream in data.get("streams", [])
        if stream.get("codec_type") == "audio"
    ]
    if not streams:
        raise RuntimeError("Aucun flux audio détecté dans ce fichier.")

    stream = streams[0]
    fmt = data.get("format", {})
    duration = _to_float(fmt.get("duration")) or _to_float(stream.get("duration"))
    bit_rate = _to_int(fmt.get("bit_rate")) or _to_int(stream.get("bit_rate"))
    bits = _to_int(stream.get("bits_per_raw_sample")) or _to_int(
        stream.get("bits_per_sample")
    )
    tags = {}
    tags.update(fmt.get("tags") or {})
    tags.update(stream.get("tags") or {})

    return {
        "path": path,
        "size": path.stat().st_size,
        "duration": duration,
        "bit_rate": bit_rate,
        "codec": str(stream.get("codec_name") or "inconnu"),
        "codec_long": str(stream.get("codec_long_name") or ""),
        "sample_fmt": str(stream.get("sample_fmt") or ""),
        "sample_rate": _to_int(stream.get("sample_rate")),
        "channels": _to_int(stream.get("channels")),
        "channel_layout": str(stream.get("channel_layout") or ""),
        "bits": bits,
        "container": str(fmt.get("format_name") or "inconnu"),
        "container_long": str(fmt.get("format_long_name") or ""),
        "tags": tags,
    }


def classify_audio(meta: dict) -> dict:
    """Détermine si le fichier est sans perte, lossy, et s'il peut aller en FLAC."""
    codec = meta["codec"].lower()
    container = meta["container"].lower()
    sample_fmt = meta["sample_fmt"].lower()
    bits = meta["bits"]
    channels = meta["channels"]
    rate = meta["sample_rate"]

    pcm = codec.startswith("pcm_")
    float_pcm = (
        codec.startswith("pcm_f")
        or sample_fmt in FLOAT_SAMPLE_FORMATS
    )
    companded = codec in {"pcm_alaw", "pcm_mulaw"} or codec.startswith("adpcm_")
    wav = "wav" in container
    aiff = "aiff" in container or "aif" in container
    flac_ok = bool(
        pcm
        and not companded
        and not float_pcm
        and bits in (8, 16, 24, 32)
        and 1 <= channels <= 8
        and 0 < rate <= 655_350
        and (wav or aiff)
    )

    if codec == "flac":
        kind = "flac"
        label = "FLAC — déjà sans perte"
    elif flac_ok:
        kind = "pcm_int"
        label = f"PCM entier {bits} bits — sans perte"
    elif pcm and not companded and float_pcm:
        kind = "pcm_float"
        label = f"PCM flottant {bits or '?'} bits — sans perte, FLAC incompatible"
    elif pcm and not companded:
        kind = "pcm_unsupported"
        label = f"PCM {bits or '?'} bits — sans perte, conversion FLAC non proposée"
    elif codec in LOSSY_CODECS or companded:
        kind = "lossy"
        label = f"{CODEC_LABELS.get(codec, codec.upper())} — format avec perte"
    else:
        kind = "other"
        label = f"{CODEC_LABELS.get(codec, codec)} — format non classé"

    return {
        "kind": kind,
        "label": label,
        "lossless": codec == "flac" or (pcm and not companded),
        "flac_ok": flac_ok,
        "pcm": pcm,
        "float_pcm": float_pcm,
        "companded": companded,
        "container_wav": wav,
        "container_aiff": aiff,
    }


def _native_pcm_bytes(meta: dict) -> int:
    """Taille approximative du PCM natif (hors en-têtes de conteneur)."""
    bits = meta["bits"]
    if bits <= 0 or meta["duration"] <= 0:
        return 0
    bytes_per_sample = (bits + 7) // 8
    return int(
        meta["duration"] * meta["sample_rate"] * meta["channels"] * bytes_per_sample
    )


# ──────────────────────────────────────────────────────────────────────────────
# Import de fichiers
# ──────────────────────────────────────────────────────────────────────────────
def _unique_target(target: Path) -> Path:
    if not target.exists():
        return target
    index = 2
    while True:
        candidate = target.with_name(f"{target.stem}_{index}{target.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def _file_key(path: Path) -> str:
    """Clé sans collision pour deux fichiers de même radical mais d'extension différente."""
    suffix = path.suffix.lower().lstrip(".") or "audio"
    return f"{path.stem}_{suffix}"


def _save_upload(uploaded) -> None:
    token = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("audio_last_upload") == token:
        return

    target = ORIGINAL_DIR / Path(uploaded.name).name
    if target.exists() and target.stat().st_size != uploaded.size:
        target = _unique_target(target)

    temporary = target.with_name(f"{target.name}.part")
    try:
        uploaded.seek(0)
        with temporary.open("wb") as output:
            shutil.copyfileobj(uploaded, output, length=1 << 20)
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    st.session_state["audio_last_upload"] = token
    st.session_state["audio_file"] = target.name
    st.sidebar.success(f"Fichier audio enregistré : {target.name}")
    st.rerun()


def _pick_file() -> Path | None:
    files = sorted(
        path for path in ORIGINAL_DIR.iterdir()
        if path.is_file() and path.suffix.lower().lstrip(".") in AUDIO_EXTENSIONS
    )
    if not files:
        return None
    names = [path.name for path in files]
    if st.session_state.get("audio_file") not in names:
        st.session_state["audio_file"] = names[0]
    selected = st.sidebar.selectbox("Fichier audio", names, key="audio_file")
    return ORIGINAL_DIR / selected


# ──────────────────────────────────────────────────────────────────────────────
# Aperçu audio et forme d'onde
# ──────────────────────────────────────────────────────────────────────────────
def _preview_path(path: Path) -> Path:
    return PREVIEW_DIR / f"{_file_key(path)}_apercu.mp3"


def _ensure_preview(path: Path) -> Path:
    target = _preview_path(path)
    if (
        target.is_file()
        and target.stat().st_size > 0
        and target.stat().st_mtime >= path.stat().st_mtime
    ):
        return target

    temporary = target.with_name(f"{target.stem}.part{target.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        _run(
            [
                FFMPEG, "-v", "error", "-nostdin", "-y",
                "-i", str(path),
                "-t", str(PREVIEW_SECONDS),
                "-map", "0:a:0", "-vn", "-sn", "-dn",
                "-ac", "2", "-ar", str(PREVIEW_RATE),
                "-c:a", "libmp3lame", "-b:a", "128k",
                str(temporary),
            ],
            timeout=600,
        )
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("ffmpeg n'a produit aucun aperçu audio.")
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _waveform_path(path: Path) -> Path:
    return PREVIEW_DIR / f"{_file_key(path)}_forme_onde.npz"


def _load_waveform(path: Path) -> dict | None:
    target = _waveform_path(path)
    if not target.is_file() or target.stat().st_mtime < path.stat().st_mtime:
        return None
    try:
        with np.load(target, allow_pickle=False) as data:
            return {
                "mins": data["mins"],
                "maxs": data["maxs"],
                "duration": float(data["duration"]),
            }
    except Exception:
        return None


def _build_waveform(path: Path, duration: float) -> dict:
    """Décode l'audio en mono 8 kHz et construit une enveloppe min/max.

    Le décodage est fait en flux : on ne garde jamais tout le PCM en mémoire.
    """
    target = _waveform_path(path)
    if duration > 0:
        bucket_size = max(1, int(round(duration * WAVEFORM_RATE / WAVEFORM_POINTS)))
    else:
        bucket_size = WAVEFORM_RATE

    cmd = [
        FFMPEG, "-v", "error", "-nostdin", "-i", str(path),
        "-map", "0:a:0", "-vn", "-sn", "-dn",
        "-ac", "1", "-ar", str(WAVEFORM_RATE),
        "-f", "s16le", "-",
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("L'outil `ffmpeg` est introuvable.") from exc
    mins: list[int] = []
    maxs: list[int] = []
    current_min = 32_767
    current_max = -32_768
    current_count = 0
    total_samples = 0
    carry = b""

    try:
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(1 << 20)
            if not chunk:
                break
            data = carry + chunk
            if len(data) % 2:
                carry = data[-1:]
                data = data[:-1]
            else:
                carry = b""
            if not data:
                continue
            samples = np.frombuffer(data, dtype="<i2")
            total_samples += int(samples.size)
            position = 0
            while position < samples.size:
                take = min(
                    int(samples.size) - position,
                    bucket_size - current_count,
                )
                segment = samples[position:position + take]
                current_min = min(current_min, int(segment.min()))
                current_max = max(current_max, int(segment.max()))
                current_count += take
                position += take
                if current_count >= bucket_size:
                    mins.append(current_min)
                    maxs.append(current_max)
                    current_min = 32_767
                    current_max = -32_768
                    current_count = 0
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        if proc.stderr is not None:
            proc.stderr.close()
        code = proc.wait()
        if code != 0:
            raise RuntimeError(stderr.strip() or f"ffmpeg a échoué (code {code}).")

    if current_count:
        mins.append(current_min)
        maxs.append(current_max)
    if not mins:
        mins = [0]
        maxs = [0]

    mins_array = np.asarray(mins, dtype=np.int16)
    maxs_array = np.asarray(maxs, dtype=np.int16)

    # Si la durée ffprobe était inconnue, on a peut-être produit plus de points
    # que nécessaire. On regroupe alors les buckets pour garder un graphe léger.
    if mins_array.size > WAVEFORM_POINTS:
        factor = int(math.ceil(mins_array.size / WAVEFORM_POINTS))
        usable = (mins_array.size // factor) * factor
        if usable:
            mins_array = mins_array[:usable].reshape(-1, factor).min(axis=1)
            maxs_array = maxs_array[:usable].reshape(-1, factor).max(axis=1)

    actual_duration = total_samples / WAVEFORM_RATE if total_samples else duration
    np.savez_compressed(
        target,
        mins=mins_array,
        maxs=maxs_array,
        duration=np.asarray(actual_duration, dtype=np.float64),
    )
    return {
        "mins": mins_array,
        "maxs": maxs_array,
        "duration": actual_duration,
    }


def _ensure_waveform(path: Path, duration: float) -> dict:
    cached = _load_waveform(path)
    if cached is not None:
        return cached
    return _build_waveform(path, duration)


def _render_waveform(path: Path, duration: float) -> None:
    try:
        with st.spinner("Analyse de la forme d'onde (décodage en flux)…"):
            waveform = _ensure_waveform(path, duration)
    except Exception as exc:
        st.warning(f"Forme d'onde indisponible : {exc}")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        st.caption(f"Graphique indisponible : {exc}")
        return

    mins = waveform["mins"].astype(np.float32) / 32768.0
    maxs = waveform["maxs"].astype(np.float32) / 32768.0
    total_duration = max(waveform["duration"], 0.001)
    times = np.linspace(0.0, total_duration, mins.size)

    figure, axis = plt.subplots(figsize=(11, 2.4))
    axis.fill_between(times, mins, maxs, color="#4c78a8", linewidth=0)
    axis.axhline(0.0, color="#888888", linewidth=0.5, alpha=0.5)
    axis.set_ylim(-1.05, 1.05)
    axis.set_xlim(0.0, total_duration)
    axis.set_xlabel("Temps (s)")
    axis.set_ylabel("Amplitude")
    axis.grid(alpha=0.18)
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True)


# ──────────────────────────────────────────────────────────────────────────────
# Encodage FLAC et vérification PCM
# ──────────────────────────────────────────────────────────────────────────────
def _flac_encode(source: Path, output: Path, level: int) -> subprocess.CompletedProcess:
    """Encode en FLAC, en conservant les chunks WAV/AIFF si possible."""
    base = [FLAC, "-f", "-s", f"-{level}"]
    output.unlink(missing_ok=True)
    try:
        proc = subprocess.run(
            [str(arg) for arg in base + [
                "--keep-foreign-metadata-if-present",
                "-o", str(output), str(source),
            ]],
            capture_output=True,
            text=True,
            timeout=7200,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("L'outil `flac` est introuvable.") from exc
    if proc.returncode == 0:
        return proc

    error = (proc.stderr or proc.stdout).strip()
    lowered = error.lower()
    if any(
        word in lowered
        for word in ("foreign", "metadata", "chunk", "unrecognized option", "unknown option")
    ):
        # Certains WAV/AIFF ont des chunks exotiques que flac ne sait pas
        # recopier. On refait l'encodage sans ces métadonnées : le PCM reste
        # strictement identique, seuls les chunks annexes sont perdus.
        proc = subprocess.run(
            [str(arg) for arg in base + ["-o", str(output), str(source)]],
            capture_output=True,
            text=True,
            timeout=7200,
        )
        if proc.returncode == 0:
            return proc
        error = (proc.stderr or proc.stdout).strip()

    output.unlink(missing_ok=True)
    raise RuntimeError(error or "flac a échoué.")


def _flac_test(path: Path) -> None:
    _run([FLAC, "-t", "-s", str(path)], timeout=7200)


def _pcm_sha256(path: Path, timeout: int = 7200) -> tuple[str, int]:
    """Décode en PCM s32le et calcule un SHA-256 en flux.

    On normalise tous les formats entiers vers s32le avec le même décodeur
    ffmpeg pour comparer les échantillons sans charger le PCM en mémoire.
    """
    cmd = [
        FFMPEG, "-v", "error", "-nostdin", "-i", str(path),
        "-map", "0:a:0", "-vn", "-sn", "-dn",
        "-acodec", "pcm_s32le", "-f", "s32le", "-",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("L'outil `ffmpeg` est introuvable.") from exc
    digest = hashlib.sha256()
    total = 0
    try:
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise RuntimeError(f"Décodage PCM trop long pour {path.name}.")
    if code != 0:
        raise RuntimeError(stderr.strip() or f"ffmpeg n'a pas pu décoder {path.name}.")
    return digest.hexdigest(), total


def _verify_flac_candidate(source: Path, output: Path, level: int) -> dict:
    """Vérifie un FLAC déjà encodé : test flac + hash PCM original/décodé."""
    started = time.perf_counter()
    _flac_test(output)
    original_hash, original_bytes = _pcm_sha256(source)
    flac_hash, flac_bytes = _pcm_sha256(output)
    verify_ms = (time.perf_counter() - started) * 1000.0
    return {
        "verified": original_hash == flac_hash and original_bytes == flac_bytes,
        "original_hash": original_hash,
        "flac_hash": flac_hash,
        "pcm_sha_bytes": original_bytes,
        "verify_ms": verify_ms,
        "level": level,
    }


def _encode_and_verify_flac(source: Path, output: Path, level: int) -> dict:
    started = time.perf_counter()
    _flac_encode(source, output, level)
    encode_ms = (time.perf_counter() - started) * 1000.0
    verification = _verify_flac_candidate(source, output, level)
    return {
        "encode_ms": encode_ms,
        **verification,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Archivage exact .rlab
# ──────────────────────────────────────────────────────────────────────────────
def _rlab_compress(source: Path, output: Path) -> dict:
    output.unlink(missing_ok=True)
    try:
        return text_lab.run_codec(
            ["compress", str(source), str(output), "lz77+huffman"],
            timeout=7200,
        )
    except Exception:
        output.unlink(missing_ok=True)
        raise


def _rlab_verify(source: Path, archive: Path) -> dict:
    text_lab.run_codec(["verify", str(source), str(archive)], timeout=7200)
    reconstructed = COMPRESSED_DIR / f"{archive.stem}_reconstruit{source.suffix or '.bin'}"
    reconstructed.unlink(missing_ok=True)
    try:
        text_lab.run_codec(
            ["decompress", str(archive), str(reconstructed)],
            timeout=7200,
        )
        same = _files_equal(source, reconstructed)
        size = reconstructed.stat().st_size
    finally:
        reconstructed.unlink(missing_ok=True)
    return {"same": same, "bytes": size}


# ──────────────────────────────────────────────────────────────────────────────
# Métriques et affichage
# ──────────────────────────────────────────────────────────────────────────────
def _render_quality_metrics(result: dict) -> None:
    st.markdown("#### Qualité de reconstruction")
    if result.get("verified"):
        columns = st.columns(4)
        columns[0].metric("MAE", "0")
        columns[1].metric("MSE", "0")
        columns[2].metric("SNR", "∞")
        columns[3].metric("PSNR", "∞")
        columns = st.columns(4)
        columns[0].metric("SSIM temps-fréquence", "1.000000")
        columns[1].metric("Erreur max", "0")
        columns[2].metric("Échantillons exacts", "100 %")
        columns[3].metric("SHA-256 PCM", "identique")
        st.caption(
            "MAE, MSE, SNR et PSNR sont ici déduits de l'identité exacte des "
            "échantillons ; SSIM serait calculée sur le spectrogramme "
            "temps-fréquence (elle n'est pas définie nativement sur une onde 1D). "
            "Puisque le hash PCM est identique, ces valeurs valent exactement "
            "0, 0, ∞, ∞ et 1. La preuve réelle reste le hash PCM calculé en flux."
        )
    else:
        st.error(
            "La vérification PCM a échoué : le FLAC ne reconstruit pas exactement "
            "les échantillons du fichier source. Le fichier n'est pas conservé."
        )


def _render_flac_result(result: dict, download_key: str) -> None:
    output_path = Path(result["output_path"])
    original_bytes = int(result["original_bytes"])
    output_bytes = int(result["output_bytes"])
    pcm_bytes = int(result.get("pcm_bytes", 0))
    gain = (original_bytes - output_bytes) / original_bytes * 100 if original_bytes else 0.0
    gain_pcm = (pcm_bytes - output_bytes) / pcm_bytes * 100 if pcm_bytes else 0.0
    output_exists = output_path.is_file()

    st.markdown("#### Résultat FLAC")
    columns = st.columns(4)
    columns[0].metric("Fichier original", fmt_bytes(original_bytes))
    columns[1].metric(
        "Fichier FLAC",
        fmt_bytes(output_bytes),
        delta=f"{gain:+.2f} %",
        help="Taille réelle du fichier FLAC, métadonnées comprises.",
    )
    columns[2].metric(
        "PCM décodé",
        fmt_bytes(pcm_bytes) if pcm_bytes else "—",
        delta=f"{gain_pcm:+.2f} % vs PCM" if pcm_bytes else None,
        help="Taille estimée des échantillons PCM natifs, hors en-tête du conteneur.",
    )
    columns[3].metric(
        "Niveau FLAC",
        f"-{result['level']}",
        help="Niveau d'encodage FLAC utilisé.",
    )

    timing = st.columns(2)
    timing[0].metric("Encodage FLAC", fmt_ms(result["encode_ms"]))
    timing[1].metric(
        "Test + hash PCM",
        fmt_ms(result["verify_ms"]),
        help="flac -t, puis décodage des deux fichiers en PCM et comparaison SHA-256.",
    )

    if result.get("oversized"):
        st.warning(
            f"Le FLAC (**{output_bytes:,} octets**) n'est pas plus petit que le "
            f"fichier original (**{original_bytes:,} octets**). Il n'est pas "
            "conservé. C'est typique d'un fichier déjà très compressé ou très "
            "court ; le PCM reste néanmoins vérifié sans perte."
        )
    elif result.get("verified"):
        st.success(
            "✅ Sans perte audio : `flac -t` est OK et le **PCM décodé est "
            "strictement identique** (SHA-256 identique). Le FLAC n'est pas "
            "octet pour octet identique au WAV/AIFF, mais les échantillons le sont."
        )
    else:
        st.error(
            "La vérification PCM a échoué. Le fichier FLAC produit est supprimé."
        )

    with st.expander("Détails de vérification", expanded=False):
        st.code(
            f"SHA-256 PCM original : {result.get('original_hash', '—')}\n"
            f"SHA-256 PCM FLAC     : {result.get('flac_hash', '—')}\n"
            f"Octets PCM s32le     : {result.get('pcm_sha_bytes', 0):,}\n"
            f"flac -t              : {'OK' if result.get('verified') else 'ÉCHEC'}"
        )

    if output_exists and result.get("verified") and not result.get("oversized"):
        st.download_button(
            "⬇️ Télécharger le FLAC",
            data=lambda path=output_path: path.open("rb"),
            file_name=output_path.name,
            mime="audio/flac",
            key=f"audio_flac_dl_{download_key}_{output_path.name}",
        )
        st.caption(
            f"Le fichier reste disponible ici : `{output_path.relative_to(ROOT)}`"
        )
    elif result.get("oversized"):
        st.caption("Aucun fichier FLAC n'est conservé, conformément à la règle "
                   "« jamais plus gros que l'original ».")


def _render_rlab_result(result: dict, download_key: str) -> None:
    stats = result["stats"]
    output_path = Path(result["output_path"])
    input_bytes = int(stats.get("input_bytes", 0))
    output_bytes = int(stats.get("output_bytes", 0))
    payload_bytes = int(stats.get("payload_bytes", 0))
    overhead = int(stats.get("container_overhead", 0))
    encode_ms = _to_float(stats.get("encode_ms"))
    decode_ms = _to_float(stats.get("decode_ms"))
    gain = (input_bytes - output_bytes) / input_bytes * 100 if input_bytes else 0.0
    gain_payload = (input_bytes - payload_bytes) / input_bytes * 100 if input_bytes else 0.0

    st.markdown("#### Résultat de l'archivage exact `.rlab`")
    columns = st.columns(4)
    columns[0].metric("Fichier original", fmt_bytes(input_bytes))
    columns[1].metric(
        "Fichier .rlab",
        fmt_bytes(output_bytes),
        delta=f"{gain:+.2f} %",
    )
    columns[2].metric(
        "Payload compressé",
        fmt_bytes(payload_bytes),
        delta=f"{gain_payload:+.2f} %",
        help="Données compressées finales, sans en-tête, comme dans compress_lab.py.",
    )
    columns[3].metric("En-tête + tables", fmt_bytes(overhead))

    timing = st.columns(2)
    timing[0].metric("Encodage Rust", fmt_ms(encode_ms))
    timing[1].metric("Décodage + vérification", fmt_ms(decode_ms))

    if result.get("oversized"):
        st.warning(
            f"Le `.rlab` (**{output_bytes:,} octets**) n'est pas plus petit que le "
            f"fichier original (**{input_bytes:,} octets**). Il n'est pas conservé. "
            "C'est attendu pour un MP3/OGG déjà compressé avec perte, un FLAC déjà "
            "optimisé ou un fichier très court : la compression générique ne peut "
            "pas faire de miracle."
        )
    elif stats.get("exact") == "1":
        st.success(
            "✅ Archivage exact : le moteur Rust a reconstruit **strictement les "
            "mêmes octets** (vérification intégrale pendant la compression)."
        )
    else:
        st.error("La vérification sans perte a échoué : ne pas utiliser ce fichier.")

    if (
        output_path.is_file()
        and not result.get("oversized")
        and stats.get("exact") == "1"
    ):
        st.download_button(
            "⬇️ Télécharger l'archive exacte (.rlab)",
            data=lambda path=output_path: path.open("rb"),
            file_name=output_path.name,
            mime="application/octet-stream",
            key=f"audio_rlab_dl_{download_key}_{output_path.name}",
        )

        verify_key = f"audio_rlab_verify_{download_key}_{output_path.name}"
        verify_button_key = f"{verify_key}_button"
        if st.button(
            "🔍 Décompresser et vérifier les octets",
            key=verify_button_key,
            help="Décompresse le .rlab et compare le fichier reconstruit à l'original.",
        ):
            with st.spinner("Décompression et comparaison octet par octet…"):
                try:
                    check = _rlab_verify(Path(result["source_path"]), output_path)
                    st.session_state[verify_key] = check
                except Exception as exc:
                    st.error(f"Vérification impossible : {exc}")
                    st.session_state.pop(verify_key, None)

        check = st.session_state.get(verify_key)
        if check:
            if check["same"]:
                st.success(
                    f"✅ Deuxième vérification réussie : {check['bytes']:,} octets "
                    "strictement identiques à l'original."
                )
            else:
                st.error("❌ Le fichier décompressé diffère de l'original.")


def _flac_comparison_rows(results: list[dict], best_level: int | None) -> pd.DataFrame:
    rows = []
    for result in results:
        verified = result.get("verified")
        if verified is True:
            verification = "✅ PCM identique"
        elif result["level"] == best_level:
            verification = "❌ échec"
        else:
            verification = "— non retenu"
        rows.append({
            "Niveau": f"-{result['level']}",
            "Taille FLAC": fmt_bytes(result["output_bytes"]),
            "Gain": f"{result['gain']:+.2f} %",
            "Encodage": fmt_ms(result["encode_ms"]),
            "Vérification": verification,
            "Meilleur": "✅" if result["level"] == best_level else "",
        })
    return pd.DataFrame(rows)


def _render_comparison(comparison: dict, download_key: str) -> None:
    st.markdown("#### Comparatif des niveaux FLAC")
    rows = comparison["rows"]
    best_level = comparison.get("best_level")
    if rows:
        st.dataframe(
            _flac_comparison_rows(rows, best_level),
            hide_index=True,
            width="stretch",
        )
    if comparison.get("errors"):
        with st.expander("Niveaux en erreur"):
            for error in comparison["errors"]:
                st.write(f"- {error}")
    best = comparison.get("best")
    if best is not None:
        _render_flac_result(best, f"{download_key}_best")
        _render_quality_metrics(best)
    else:
        st.error("Aucun niveau FLAC n'a pu être produit.")
        if not comparison.get("errors"):
            st.caption("Les trois encodages ont été refusés ou supprimés.")


# ──────────────────────────────────────────────────────────────────────────────
# Actions
# ──────────────────────────────────────────────────────────────────────────────
def _run_single_flac(path: Path, meta: dict, level: int) -> dict:
    output_path = COMPRESSED_DIR / f"{_file_key(path)}_flac{level}.flac"
    original_bytes = path.stat().st_size
    pcm_bytes = _native_pcm_bytes(meta)

    # Un comparatif précédent a pu laisser ses fichiers internes : on les
    # supprime pour ne pas les confondre avec le résultat courant.
    for comparison_level in FLAC_LEVELS:
        (COMPRESSED_DIR / f"{_file_key(path)}_flac{comparison_level}_cmp.flac").unlink(
            missing_ok=True
        )

    try:
        with st.spinner(f"Encodage FLAC niveau -{level}…"):
            verification = _encode_and_verify_flac(path, output_path, level)
    except Exception:
        output_path.unlink(missing_ok=True)
        raise

    output_bytes = output_path.stat().st_size if output_path.is_file() else 0
    oversized = output_bytes >= original_bytes
    if not verification["verified"] or oversized:
        output_path.unlink(missing_ok=True)

    return {
        "kind": "flac",
        "signature": (str(path), path.stat().st_mtime_ns, path.stat().st_size, level),
        "source_path": str(path),
        "output_path": str(output_path),
        "original_bytes": original_bytes,
        "output_bytes": output_bytes,
        "pcm_bytes": pcm_bytes,
        "oversized": oversized,
        **verification,
    }


def _run_flac_comparison(path: Path, meta: dict) -> dict:
    original_bytes = path.stat().st_size
    pcm_bytes = _native_pcm_bytes(meta)
    signature = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
    candidates: list[tuple[int, Path, float]] = []
    errors: list[str] = []

    # Supprime les éventuels candidats d'un comparatif précédent, même si
    # l'encodage échoue avant de les avoir tous réécrits.
    for level in FLAC_LEVELS:
        (COMPRESSED_DIR / f"{_file_key(path)}_flac{level}_cmp.flac").unlink(
            missing_ok=True
        )

    with st.spinner("Comparaison des niveaux FLAC -0 / -5 / -8…"):
        for level in FLAC_LEVELS:
            output_path = COMPRESSED_DIR / f"{_file_key(path)}_flac{level}_cmp.flac"
            started = time.perf_counter()
            try:
                _flac_encode(path, output_path, level)
                if not output_path.is_file():
                    raise RuntimeError("flac n'a produit aucun fichier.")
                encode_ms = (time.perf_counter() - started) * 1000.0
                candidates.append((level, output_path, encode_ms))
            except Exception as exc:
                output_path.unlink(missing_ok=True)
                errors.append(f"Niveau -{level} : {exc}")

        if not candidates:
            return {
                "signature": signature,
                "rows": [],
                "errors": errors,
                "best": None,
                "best_level": None,
            }

        try:
            candidates.sort(key=lambda item: item[1].stat().st_size)
            best_level, best_path, best_encode_ms = candidates[0]
        except OSError as exc:
            for _, candidate_path, _ in candidates:
                candidate_path.unlink(missing_ok=True)
            return {
                "signature": signature,
                "rows": [],
                "errors": errors + [f"Lecture des fichiers FLAC : {exc}"],
                "best": None,
                "best_level": None,
            }

        # On ne vérifie en détail que le gagnant.
        try:
            verification = _verify_flac_candidate(path, best_path, best_level)
        except Exception as exc:
            for _, candidate_path, _ in candidates:
                candidate_path.unlink(missing_ok=True)
            return {
                "signature": signature,
                "rows": [],
                "errors": errors + [f"Vérification du niveau -{best_level} : {exc}"],
                "best": None,
                "best_level": None,
            }

        rows = []
        for level, candidate_path, encode_ms in candidates:
            candidate_bytes = candidate_path.stat().st_size
            rows.append({
                "level": level,
                "output_bytes": candidate_bytes,
                "gain": (original_bytes - candidate_bytes) / original_bytes * 100
                        if original_bytes else 0.0,
                "encode_ms": encode_ms,
                "verified": verification["verified"] if level == best_level else None,
            })

        best = {
            "kind": "flac",
            "signature": (
                str(path), path.stat().st_mtime_ns, path.stat().st_size, "compare",
            ),
            "source_path": str(path),
            "output_path": str(best_path),
            "original_bytes": original_bytes,
            "output_bytes": best_path.stat().st_size,
            "pcm_bytes": pcm_bytes,
            "encode_ms": best_encode_ms,
            "oversized": best_path.stat().st_size >= original_bytes,
            **verification,
        }

        for level, candidate_path, _ in candidates:
            if candidate_path != best_path:
                candidate_path.unlink(missing_ok=True)
        if not verification["verified"] or best["oversized"]:
            best_path.unlink(missing_ok=True)

    return {
        "signature": signature,
        "rows": rows,
        "errors": errors,
        "best": best,
        "best_level": best_level,
    }


def _run_rlab(path: Path, meta: dict) -> dict:
    output_path = COMPRESSED_DIR / f"{_file_key(path)}_exact.rlab"
    input_bytes = path.stat().st_size
    try:
        with st.spinner("Archivage exact avec le moteur Rust (LZ77 + Huffman)…"):
            stats = _rlab_compress(path, output_path)
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    output_bytes = output_path.stat().st_size if output_path.is_file() else 0
    oversized = output_bytes >= input_bytes
    exact = stats.get("exact") == "1"
    if oversized or not exact:
        output_path.unlink(missing_ok=True)
    return {
        "kind": "rlab",
        "signature": (
            str(path), path.stat().st_mtime_ns, path.stat().st_size, "rlab",
        ),
        "source_path": str(path),
        "output_path": str(output_path),
        "stats": stats,
        "oversized": oversized,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Interface principale
# ──────────────────────────────────────────────────────────────────────────────
def _render_metadata(path: Path, meta: dict, classification: dict) -> None:
    st.markdown(f"### `{path.name}`")
    if classification["lossless"]:
        st.success(f"✅ {classification['label']}")
    else:
        st.warning(f"⚠️ {classification['label']}")

    columns = st.columns(4)
    columns[0].metric("Taille", fmt_bytes(meta["size"]))
    columns[1].metric("Durée", fmt_duration(meta["duration"]))
    columns[2].metric(
        "Fréquence",
        f"{meta['sample_rate'] / 1000:.1f} kHz" if meta["sample_rate"] else "—",
    )
    columns[3].metric(
        "Canaux",
        f"{meta['channels']}" if meta["channels"] else "—",
        help=meta["channel_layout"] or None,
    )
    columns = st.columns(4)
    columns[0].metric("Codec", meta["codec"])
    columns[1].metric(
        "Profondeur",
        f"{meta['bits']} bits" if meta["bits"] else "—",
    )
    columns[2].metric(
        "Débit",
        f"{meta['bit_rate'] / 1000:.0f} kb/s" if meta["bit_rate"] else "—",
    )
    columns[3].metric(
        "Conteneur",
        meta["container"].split(",")[0] if meta["container"] else "—",
    )

    with st.expander("Détails techniques et tags", expanded=False):
        st.json({
            "codec": meta["codec"],
            "codec_long": meta["codec_long"],
            "sample_fmt": meta["sample_fmt"],
            "sample_rate": meta["sample_rate"],
            "channels": meta["channels"],
            "channel_layout": meta["channel_layout"],
            "bits": meta["bits"],
            "container": meta["container"],
            "container_long": meta["container_long"],
            "tags": meta["tags"],
        })


def _render_preview(path: Path, meta: dict) -> None:
    try:
        preview = _ensure_preview(path)
        st.audio(preview.read_bytes(), format="audio/mp3")
        st.caption(
            "Aperçu de 30 s en MP3 128 kb/s, uniquement pour l'écoute. "
            "Il ne participe pas à la compression et n'est pas conservé comme résultat."
        )
    except Exception as exc:
        st.warning(f"Aperçu audio indisponible : {exc}")


def _render_lossy_notice(classification: dict) -> None:
    if classification["kind"] != "lossy":
        return
    st.warning(
        "**Format avec perte.** Le fichier MP3/OGG (ou autre codec avec perte) "
        "contient déjà une version dégradée du signal : le décoder ne retrouve "
        "pas l'original. On peut seulement conserver le fichier **octet pour "
        "octet** dans un `.rlab`. On ne peut pas le compresser davantage sans "
        "perte, ni restaurer la qualité perdue. Les métriques MAE/SNR/PSNR/SSIM "
        "ne sont pas mesurables sans le master non compressé ; elles ne seraient "
        "pas informatives ici."
    )


def _render_float_notice(classification: dict) -> None:
    if classification["kind"] == "pcm_float":
        st.info(
            "FLAC ne code que le PCM entier. Ce fichier est sans perte à la "
            "source, mais il ne peut pas être converti en FLAC sans passer par "
            "du PCM entier (ce qui serait avec perte). On propose donc un "
            "archivage exact `.rlab`."
        )
    elif classification["kind"] == "pcm_unsupported":
        st.info(
            "Ce PCM entier n'entre pas dans les paramètres FLAC proposés "
            "(profondeur, nombre de canaux ou fréquence). Plutôt qu'une "
            "conversion risquée, on propose l'archivage exact `.rlab`."
        )


def _render_flac_controls(path: Path, meta: dict, classification: dict) -> None:
    st.markdown("#### Compression FLAC sans perte")
    st.caption(
        "FLAC est un codec spécialisé audio sans perte. Le PCM reconstruit est "
        "vérifié par `flac -t` puis par comparaison SHA-256 du PCM décodé."
    )
    level = st.selectbox(
        "Niveau FLAC",
        options=FLAC_LEVELS,
        index=1,
        format_func=lambda value: FLAC_LEVEL_LABELS[value],
        key=f"audio_flac_level_{path.name}",
        help="0 = très rapide, 5 = bon compromis, 8 = plus petit mais plus lent. "
             "Aucune limite de taille n'est appliquée.",
    )
    st.caption(
        "Pour un test rapide, choisis **-0**. Pour optimiser, choisis **-8** ou "
        "lance le comparatif des trois niveaux."
    )

    signature = (str(path), path.stat().st_mtime_ns, path.stat().st_size, level)
    run = st.button(
        "🚀 Compresser en FLAC sans perte",
        type="primary",
        key=f"audio_flac_run_{path.name}",
    )
    compare = st.button(
        "🔬 Comparer les niveaux -0 / -5 / -8",
        key=f"audio_flac_compare_{path.name}",
        help="Encode trois fois et conserve le plus petit. Plus lent, surtout sur "
             "un gros fichier. À utiliser quand tu veux optimiser au maximum.",
    )

    if run:
        try:
            result = _run_single_flac(path, meta, level)
            st.session_state["audio_flac_result"] = result
            st.session_state.pop("audio_flac_compare_result", None)
        except Exception as exc:
            st.error(f"Compression FLAC impossible : {exc}")
            st.session_state.pop("audio_flac_result", None)
            st.session_state.pop("audio_flac_compare_result", None)

    if compare:
        try:
            comparison = _run_flac_comparison(path, meta)
            st.session_state["audio_flac_compare_result"] = comparison
            st.session_state.pop("audio_flac_result", None)
        except Exception as exc:
            st.error(f"Comparaison FLAC impossible : {exc}")
            st.session_state.pop("audio_flac_compare_result", None)
            st.session_state.pop("audio_flac_result", None)

    stored = st.session_state.get("audio_flac_result")
    if stored is not None:
        if stored["signature"] == signature:
            _render_flac_result(stored, path.stem)
            _render_quality_metrics(stored)
        else:
            st.info("La configuration a changé depuis le dernier test : relance la compression.")

    comparison = st.session_state.get("audio_flac_compare_result")
    if comparison is not None:
        expected = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        if comparison.get("signature") == expected:
            _render_comparison(comparison, path.stem)
        else:
            st.info("Le fichier a changé depuis le dernier comparatif : relance-le.")

    with st.expander("Archiver aussi le WAV/AIFF original en .rlab (octets exacts)"):
        st.caption(
            "Le FLAC garantit le PCM, pas l'identité octet pour octet du conteneur. "
            "Cette option conserve le fichier original tel quel, mais elle est "
            "généralement plus lente et plus grosse qu'un FLAC. Le moteur Rust "
            "charge le fichier en mémoire : sur un très gros WAV, préfère FLAC."
        )
        _render_rlab_controls(
            path, meta, classification,
            "Archivage exact du conteneur WAV/AIFF d'origine.",
        )


def _render_rlab_controls(
    path: Path,
    meta: dict,
    classification: dict,
    reason: str,
) -> None:
    st.markdown("#### Archivage exact `.rlab`")
    st.caption(reason)
    st.caption(
        "Le moteur Rust compresse les **octets du fichier** avec LZ77 + Huffman, "
        "puis vérifie la reconstruction. Sur un MP3/OGG/FLAC déjà compressé, le "
        "`.rlab` sera souvent plus gros : il sera alors supprimé."
    )
    signature = (
        str(path), path.stat().st_mtime_ns, path.stat().st_size, "rlab",
    )
    if st.button(
        "🚀 Archiver exactement en .rlab",
        type="primary",
        key=f"audio_rlab_run_{path.name}",
    ):
        try:
            result = _run_rlab(path, meta)
            st.session_state["audio_rlab_result"] = result
        except Exception as exc:
            st.error(f"Archivage .rlab impossible : {exc}")
            st.session_state.pop("audio_rlab_result", None)

    stored = st.session_state.get("audio_rlab_result")
    if stored is not None:
        if stored["signature"] == signature:
            _render_rlab_result(stored, path.stem)
            if classification["kind"] == "lossy":
                st.info(
                    "Rappel : l'archivage exact conserve le MP3/OGG tel quel. "
                    "Il n'améliore pas la qualité audio et ne restaure aucune "
                    "information perdue."
                )
        else:
            st.info("La configuration a changé depuis le dernier test : relance l'archivage.")


def render_audio_lab() -> None:
    st.subheader("🎵 Compression audio — sans perte")
    st.caption(
        "WAV/AIFF PCM → FLAC sans perte · FLAC → archivage exact · "
        "MP3/OGG → archivage exact avec avertissement « format avec perte » · "
        "vérification PCM ou octet par octet."
    )

    st.sidebar.header("🎵 Fichier audio")
    uploaded = st.sidebar.file_uploader(
        "Importer un fichier audio",
        type=AUDIO_EXTENSIONS,
        key="audio_upload",
        help="WAV, FLAC, AIFF/AIF, MP3, OGG/Opus. Le format réel est détecté "
             "avec ffprobe, pas seulement l'extension.",
    )
    if uploaded is not None:
        try:
            _save_upload(uploaded)
        except Exception as exc:
            st.sidebar.error(f"Impossible d'enregistrer le fichier : {exc}")

    st.sidebar.caption(
        "Outils utilisés : `ffmpeg`, `ffprobe` et `flac`. Pour un fichier plus "
        "gros que la limite d'import du navigateur, copie-le directement dans "
        "`audios/original/`."
    )
    if st.sidebar.button("🔄 Rafraîchir la liste", key="audio_refresh"):
        st.rerun()

    path = _pick_file()
    if path is None:
        st.info(
            "Aucun fichier audio trouvé. Importe un WAV, FLAC, AIFF, MP3 ou OGG, "
            "ou copie-le dans `audios/original/`."
        )
        st.code("mkdir -p audios/original audios/compressed\ncp ~/Musique/mon_audio.wav audios/original/")
        return

    try:
        meta = probe_audio(path)
    except Exception as exc:
        st.error(f"Lecture audio impossible : {exc}")
        return

    classification = classify_audio(meta)
    _render_metadata(path, meta, classification)

    preview_column, note_column = st.columns([2, 1])
    with preview_column:
        _render_preview(path, meta)
        _render_waveform(path, meta["duration"])
    with note_column:
        if classification["flac_ok"]:
            st.success(
                "Source sans perte. La compression FLAC doit rester sans perte."
            )
        elif classification["kind"] == "flac":
            st.success(
                "FLAC est déjà sans perte. L'archivage exact conserve les octets."
            )
        elif classification["lossless"]:
            st.info(
                "Source sans perte, mais FLAC n'est pas applicable ici : on "
                "propose l'archivage exact des octets."
            )
        elif classification["kind"] == "lossy":
            st.warning(
                "Source avec perte. Aucun algorithme ne peut restaurer le signal "
                "d'origine ; on peut seulement archiver les octets."
            )
        else:
            st.info(
                "Format non classé comme sans perte. On propose l'archivage exact "
                "des octets, sans prétendre améliorer la qualité."
            )
        st.caption(
            "Le PCM n'est jamais chargé entièrement : les outils décodent en flux "
            "et les vérifications utilisent des hash."
        )

    st.divider()
    _render_lossy_notice(classification)
    _render_float_notice(classification)

    if classification["flac_ok"]:
        _render_flac_controls(path, meta, classification)
    elif classification["kind"] == "flac":
        st.caption(
            "FLAC est sans perte par construction : MAE et MSE seraient nulles, "
            "SNR et PSNR infinies, et SSIM vaudrait 1. Aucun ré-encodage n'étant "
            "effectué ici, ces métriques ne sont pas recalculées ; c'est le "
            "fichier lui-même qui est archivé."
        )
        _render_rlab_controls(
            path, meta, classification,
            "FLAC est déjà un format sans perte. Un ré-encodage FLAC apporterait "
            "très peu et pourrait perdre des tags ; on propose donc l'archivage "
            "exact du fichier.",
        )
    elif classification["kind"] == "lossy":
        _render_rlab_controls(
            path, meta, classification,
            "Format avec perte : on conserve le fichier exactement, sans promettre "
            "de gain de qualité ni de taille.",
        )
    else:
        _render_rlab_controls(
            path, meta, classification,
            "Ce format n'est pas encodable directement en FLAC par l'outil installé. "
            "L'archivage exact garantit au moins les octets.",
        )

    st.divider()
    st.caption(
        "Le FLAC vérifie le **PCM** (échantillons), pas l'identité octet pour octet "
        "du conteneur WAV/AIFF. Pour une conservation fichier stricte, utilise "
        "l'archivage `.rlab`. Aucun résultat plus gros que l'original n'est conservé. "
        "Les métriques MAE/MSE/SNR/PSNR/SSIM sont affichées pour une reconstruction "
        "sans perte ; pour le MP3/OGG, seule l'intégrité bit à bit est mesurable."
    )
