#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPRESS-LAB FULL  ·  BWT · RLE · Huffman · Arithmétique · LZ77 · LZ78 · LZW
- Pipelines purs (solo) et toutes les combinaisons ordonnées
  RLE / BWT / LZ77 / LZ78 / LZW, avec Huffman ou Arithmétique en terminal
- Choix libre de la source : fichier, stdin, argument, ou texte par défaut
- Mesure du payload uniquement
- Démarche détaillée du meilleur pipeline + journal de vérification
- Mesure précise + vérification sans perte
"""

import argparse
import hashlib
import heapq
import os
import re
import sys
import time
import unicodedata
from bisect import bisect_right
from collections import Counter
from typing import List, Tuple, Dict, Any

DEFAULT_TEXT = """La compression sans perte réduit la taille d'un texte tout en permettant de le reconstruire exactement.
Ce banc d'essai compare RLE, BWT, LZ77, LZ78, LZW, Huffman et le codage arithmétique, seuls ou combinés."""

class Ansi:
    enabled = False

class Animate:
    enabled = False

def paint(text, *codes):
    if not Ansi.enabled:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"

def animate_print(text, color="36", delay=0.006):
    """Petit effet de révélation, uniquement dans un terminal interactif."""
    if not Animate.enabled:
        print(paint(text, "1", color) if Ansi.enabled else text)
        return
    for ch in text:
        sys.stdout.write(paint(ch, "1", color))
        sys.stdout.flush()
        time.sleep(delay)
    print()

def tag(kind, msg):
    colors = {"*": "36", "+": "32", "!": "33", "x": "31"}
    return f"{paint(f'[{kind}]', '1', colors.get(kind, '37'))} {msg}"

BANNER = r"""
  ┌──────────────────────────────────────────────────────────────────────┐
  │  COMPRESS-LAB  ·  BWT · RLE · Huffman · Arith · LZ77 · LZ78 · LZW    │
  │  Pipelines purs + mixtes · Source libre · Sans perte                 │
  └──────────────────────────────────────────────────────────────────────┘"""

def trunc_dec(num, den, d=4):
    if den == 0:
        return "inf"
    scale = 10 ** d
    n = (num * scale) // den
    if n or num == 0:
        return f"{n // scale},{n % scale:0{d}d}"
    e = max(1, ((den.bit_length() - num.bit_length()) * 30103) // 100000)
    while num * 10 ** e < den:
        e += 1
    while e > 1 and num * 10 ** (e - 1) >= den:
        e -= 1
    m = (num * 10 ** (e + d)) // den
    return f"{m // scale},{m % scale:0{d}d}e-{e}"

def gain_str(raw, total, d=4):
    if raw >= total:
        return trunc_dec((raw - total) * 100, raw, d)
    return "-" + trunc_dec((total - raw) * 100, raw, d)

def bits_for(x: int) -> int:
    return max(1, int(x).bit_length()) if x >= 0 else 1

def sym_bits(s) -> int:
    if isinstance(s, str):
        return 8 * max(1, len(s.encode("utf-8")))
    if isinstance(s, (list, tuple)):
        return 8 + sum(sym_bits(x) for x in s)
    if isinstance(s, int):
        return max(1, (abs(s).bit_length() + 7) // 8) * 8
    return 32

def render_table(title, headers, rows, aligns):
    real = [r for r in rows if r is not None]
    widths = [max([len(headers[i])] + [len(str(r[i])) for r in real])
              for i in range(len(headers))]

    def line(l, m, r):
        return paint(l + m.join("─" * (w + 2) for w in widths) + r, "2", "32")

    def pad(v, i):
        s = str(v)
        if aligns[i] == "right":
            return s.rjust(widths[i])
        if aligns[i] == "center":
            return s.center(widths[i])
        return s.ljust(widths[i])

    bar = paint("│", "2", "32")
    print(f"\n{tag('+', title)}")
    print(line("┌", "┬", "┐"))
    print(bar + bar.join(" " + paint(h.center(widths[i]), "1", "32") + " " for i, h in enumerate(headers)) + bar)
    print(line("├", "┼", "┤"))
    for r in rows:
        if r is None:
            print(bar + bar.join(" " + paint("⋮".center(w), "2") + " " for w in widths) + bar)
        else:
            print(bar + bar.join(f" {pad(v, i)} " for i, v in enumerate(r)) + bar)
    print(line("└", "┴", "┘"))

def canonical(table: List[Tuple[Any, int]]) -> Dict[Any, str]:
    if not table:
        return {}
    code, prev, out = 0, table[0][1], {}
    for s, L in table:
        code <<= (L - prev)
        prev = L
        out[s] = format(code, f"0{L}b")
        code += 1
    return out

class Huffman:
    label = "Huffman"

    @staticmethod
    def enc(stream: List[Any]) -> Dict[str, Any]:
        if not stream:
            return {"table": [], "bits": "", "payload": 0}
        counts = Counter(stream)
        order = {s: i for i, s in enumerate(dict.fromkeys(stream))}
        heap = [(c, order[s], ("L", s)) for s, c in counts.items()]
        heapq.heapify(heap)
        t = len(order)
        while len(heap) > 1:
            a = heapq.heappop(heap)
            b = heapq.heappop(heap)
            heapq.heappush(heap, (a[0] + b[0], t, ("N", a[2], b[2])))
            t += 1
        lengths = {}
        stack = [(heap[0][2], 0)]
        while stack:
            node, depth = stack.pop()
            if node[0] == "L":
                lengths[node[1]] = max(depth, 1)
            else:
                stack.append((node[1], depth + 1))
                stack.append((node[2], depth + 1))
        table = sorted(lengths.items(), key=lambda kv: (kv[1], order[kv[0]]))
        codes = canonical(table)
        bits = "".join(codes[s] for s in stream)
        return {"table": table, "bits": bits, "payload": len(bits)}

    @staticmethod
    def dec(b: Dict[str, Any]) -> List[Any]:
        if not b["table"]:
            return []
        inv = {c: s for s, c in canonical(b["table"]).items()}
        out, cur = [], ""
        for ch in b["bits"]:
            cur += ch
            if cur in inv:
                out.append(inv[cur])
                cur = ""
        return out

class Arith:
    label = "Arithmétique"

    @staticmethod
    def enc(stream: List[Any]) -> Dict[str, Any]:
        N = len(stream)
        if N == 0:
            return {"k": 0, "m": 0, "N": 0, "syms": [], "counts": [], "payload": 0}
        counts = Counter(stream)
        syms = list(dict.fromkeys(stream))
        cum, acc = {}, 0
        for s in syms:
            cum[s] = acc
            acc += counts[s]
        Ln, Rn, den = 0, 1, 1
        for s in stream:
            Ln = Ln * N + Rn * cum[s]
            Rn *= counts[s]
            den *= N

        def fits(k):
            scale = 1 << k
            left_scaled = Ln * scale
            right_scaled = (Ln + Rn) * scale
            mid_num = 2 * Ln + Rn
            mid_den = 2 * den
            base = (mid_num * scale) // mid_den
            best = None
            for m in (base, base + 1):
                if left_scaled < m * den < right_scaled:
                    distance = abs(m * mid_den - mid_num * scale)
                    if best is None or distance < best[0]:
                        best = (distance, m)
            if best is None:
                return False, 0
            return True, best[1]

        lo, hi = 0, max(1, den.bit_length() - Rn.bit_length() + 2)
        while not fits(hi)[0]:
            hi *= 2
        while lo < hi:
            mid = (lo + hi) // 2
            if fits(mid)[0]:
                hi = mid
            else:
                lo = mid + 1
        k = lo
        return {
            "k": k, "m": fits(k)[1], "N": N, "syms": syms,
            "counts": [counts[s] for s in syms],
            "payload": k,
        }

    @staticmethod
    def dec(b: Dict[str, Any]) -> List[Any]:
        N, syms, cnt = b["N"], b["syms"], b["counts"]
        if N == 0:
            return []
        cum = [0]
        for c in cnt:
            cum.append(cum[-1] + c)
        a, d, out = b["m"], 1 << b["k"], []
        for _ in range(N):
            t = (a * N) // d
            i = bisect_right(cum, t) - 1
            out.append(syms[i])
            a = a * N - cum[i] * d
            d *= cnt[i]
        return out

class _NoSymbol:
    def __repr__(self):
        return "<fin>"


NO_SYMBOL = _NoSymbol()


def sorted_alphabet(seq: List[Any]) -> List[Any]:
    """Ordre stable pour BWT, même avec des symboles hétérogènes."""
    uniq = list(set(seq))
    try:
        return sorted(uniq)
    except TypeError:
        return sorted(uniq, key=repr)


def bwt_seq_encode(seq: List[Any]):
    """BWT pur : dernière colonne + index primaire, sans MTF ni ZLE."""
    n = len(seq)
    if n == 0:
        return [], {"idx": 0, "alpha": []}, 0

    alpha = sorted_alphabet(seq)
    rank = {s: i for i, s in enumerate(alpha)}
    ids = [rank[s] for s in seq]
    ss = ids + ids
    rows = sorted(range(n), key=lambda i: ss[i:i + n])
    last = [seq[(i - 1) % n] for i in rows]
    idx = rows.index(0)

    plain_bits = sum(sym_bits(s) for s in last)
    return last, {"idx": idx, "alpha": alpha}, plain_bits


def bwt_seq_decode(seq: List[Any], meta: Dict[str, Any]) -> List[Any]:
    last = seq
    n = len(last)
    if n == 0:
        return []
    alpha = meta["alpha"]
    rank = {s: i for i, s in enumerate(alpha)}
    nxt = sorted(range(n), key=lambda i: rank[last[i]])
    first = [last[i] for i in nxt]
    out, row = [], meta["idx"]
    for _ in range(n):
        out.append(first[row])
        row = nxt[row]
    return out


def rle_seq_encode(seq: List[Any]):
    """RLE générique : séquence aplatie [symbole, compteur, ...]."""
    if not seq:
        return [], {}, 0
    runs = []
    prev, cnt = seq[0], 1
    for sym in seq[1:]:
        if sym == prev:
            cnt += 1
        else:
            runs.append((prev, cnt))
            prev, cnt = sym, 1
    runs.append((prev, cnt))
    out = []
    for sym, cnt in runs:
        out.append(sym)
        out.append(cnt)
    plain_bits = sum(sym_bits(sym) + bits_for(cnt) for sym, cnt in runs)
    return out, {}, plain_bits


def rle_seq_decode(seq: List[Any], meta: Dict[str, Any]) -> List[Any]:
    if len(seq) % 2:
        raise ValueError("flux RLE impair")
    out = []
    for i in range(0, len(seq), 2):
        out.extend([seq[i]] * int(seq[i + 1]))
    return out


def lz77_seq_encode(seq: List[Any], window: int = 4096, lookahead: int = 32,
                    max_chain: int = 64) -> List[Tuple[Any, ...]]:
    n = len(seq)
    if n == 0:
        return []
    positions = {}

    def remember(idx):
        if idx + 2 < n:
            key = (seq[idx], seq[idx + 1], seq[idx + 2])
            positions.setdefault(key, []).append(idx)

    tokens, i = [], 0
    while i < n:
        best_len, best_dist = 0, 0
        if i + 2 < n:
            key = (seq[i], seq[i + 1], seq[i + 2])
            cands = positions.get(key, ())
            checked = 0
            for j in reversed(cands):
                dist = i - j
                if dist > window:
                    break
                checked += 1
                if checked > max_chain:
                    break
                max_len = min(lookahead, n - i, dist)
                length = 0
                while length < max_len and seq[j + length] == seq[i + length]:
                    length += 1
                if length > best_len:
                    best_len, best_dist = length, dist
                    if best_len >= lookahead:
                        break

        if best_len >= 3:
            nxt = seq[i + best_len] if i + best_len < n else NO_SYMBOL
            tokens.append(('M', best_dist, best_len, nxt))
            step = best_len + (0 if nxt is NO_SYMBOL else 1)
            for k in range(step):
                remember(i + k)
            i += step
        else:
            tokens.append(('L', seq[i]))
            remember(i)
            i += 1
    return tokens


def lz77_seq_decode(tokens: List[Tuple[Any, ...]]) -> List[Any]:
    out = []
    for tok in tokens:
        if tok[0] == 'L':
            out.append(tok[1])
        elif tok[0] == 'M':
            dist, length = tok[1], tok[2]
            nxt = tok[3] if len(tok) > 3 else NO_SYMBOL
            start = len(out) - dist
            for k in range(length):
                out.append(out[start + k])
            if nxt is not NO_SYMBOL:
                out.append(nxt)
    return out


def estimate_lz77_seq_bits(tokens: List[Tuple[Any, ...]]) -> int:
    bits = 0
    for tok in tokens:
        if tok[0] == 'L':
            bits += 1 + sym_bits(tok[1])
        else:
            bits += 1 + bits_for(tok[1]) + bits_for(tok[2])
            if tok[3] is not NO_SYMBOL:
                bits += sym_bits(tok[3])
    return bits


def lz78_seq_encode(seq: List[Any]) -> List[Tuple[Any, ...]]:
    dictionary = {(): 0}
    next_idx, w, tokens = 1, (), []
    for sym in seq:
        wc = w + (sym,)
        if wc in dictionary:
            w = wc
        else:
            tokens.append(('D', dictionary[w], sym))
            dictionary[wc] = next_idx
            next_idx += 1
            w = ()
    if w:
        tokens.append(('D', dictionary[w], NO_SYMBOL))
    return tokens


def lz78_seq_decode(tokens: List[Tuple[Any, ...]]) -> List[Any]:
    dictionary = {0: ()}
    next_idx, out = 1, []
    for tok in tokens:
        idx, sym = tok[1], tok[2]
        entry = dictionary[idx]
        if sym is not NO_SYMBOL:
            entry = entry + (sym,)
        out.extend(entry)
        dictionary[next_idx] = entry
        next_idx += 1
    return out


def estimate_lz78_seq_bits(tokens: List[Tuple[Any, ...]]) -> int:
    bits, max_idx = 0, 0
    for tok in tokens:
        idx, sym = tok[1], tok[2]
        max_idx = max(max_idx, idx)
        bits += bits_for(idx) + (1 if sym is NO_SYMBOL else sym_bits(sym))
    return bits + bits_for(max_idx) * 2


def lz77_stage_encode(seq: List[Any]):
    tokens = lz77_seq_encode(seq)
    return tokens, {}, estimate_lz77_seq_bits(tokens)


def lz78_stage_encode(seq: List[Any]):
    tokens = lz78_seq_encode(seq)
    return tokens, {}, estimate_lz78_seq_bits(tokens)


def lzw_alphabet(seq: List[Any]) -> List[Any]:
    """Symboles distincts, dans leur ordre d'apparition."""
    alphabet, seen = [], set()
    for sym in seq:
        if sym not in seen:
            seen.add(sym)
            alphabet.append(sym)
    return alphabet


def lzw_seq_encode(seq: List[Any]):
    """LZW générique : dictionnaire indexé à partir de 1."""
    alphabet = lzw_alphabet(seq)
    dictionary = {(sym,): idx for idx, sym in enumerate(alphabet, 1)}
    next_idx, w, codes = len(alphabet) + 1, (), []
    for sym in seq:
        wc = w + (sym,)
        if wc in dictionary:
            w = wc
        else:
            codes.append(dictionary[w])
            dictionary[wc] = next_idx
            next_idx += 1
            w = (sym,)
    if w:
        codes.append(dictionary[w])
    return codes, {"alphabet": alphabet}


def lzw_seq_decode(codes: List[int], meta: Dict[str, Any]) -> List[Any]:
    alphabet = meta["alphabet"]
    dictionary = {idx: (sym,) for idx, sym in enumerate(alphabet, 1)}
    next_idx, out, previous = len(alphabet) + 1, [], None
    for code in codes:
        if code in dictionary:
            entry = dictionary[code]
        elif previous is not None and code == next_idx:
            entry = previous + (previous[0],)
        else:
            raise ValueError(f"code LZW invalide : {code}")
        out.extend(entry)
        if previous is not None:
            dictionary[next_idx] = previous + (entry[0],)
            next_idx += 1
        previous = entry
    return out


def estimate_lzw_seq_bits(codes: List[int], alphabet: List[Any]) -> int:
    if not codes:
        return 0
    width = bits_for(max(codes))
    return len(codes) * width + sum(sym_bits(sym) for sym in alphabet)


def lzw_stage_encode(seq: List[Any]):
    codes, meta = lzw_seq_encode(seq)
    return codes, meta, estimate_lzw_seq_bits(codes, meta["alphabet"])


STAGE_RLE = {"name": "RLE", "enc": rle_seq_encode, "dec": rle_seq_decode}
STAGE_BWT = {"name": "BWT", "enc": bwt_seq_encode, "dec": bwt_seq_decode}
STAGE_LZ77 = {"name": "LZ77", "enc": lz77_stage_encode, "dec": lambda s, m: lz77_seq_decode(s)}
STAGE_LZ78 = {"name": "LZ78", "enc": lz78_stage_encode, "dec": lambda s, m: lz78_seq_decode(s)}
STAGE_LZW = {"name": "LZW", "enc": lzw_stage_encode, "dec": lambda s, m: lzw_seq_decode(s, m)}
STAGES = [STAGE_RLE, STAGE_BWT, STAGE_LZ77, STAGE_LZ78, STAGE_LZW]
STAGE_BY_NAME = {stage["name"]: stage for stage in STAGES}
CODER_MAP = {Huffman.label: Huffman, Arith.label: Arith}


def run_all_pipelines(text: str, max_stages: int = 4,
                      stages=None) -> List[Dict[str, Any]]:
    """Toutes les sous-séquences ordonnées des préprocesseurs choisis,
    suivies ou non de Huffman/Arithmétique (codeurs terminaux)."""
    raw_bits = len(text.encode("utf-8")) * 8
    target = list(text)
    stage_list = list(STAGES if stages is None else stages)
    results = []
    failure_payload = raw_bits * 100 + 1

    def add(label: str, payload: int, ok: bool, category: str = "",
            stages=None, coder=None):
        results.append({
            "label": label,
            "payload": payload,
            "ok": ok,
            "category": category,
            "stages": list(stages or []),
            "coder": coder,
        })

    def verify(stream, dec_meta):
        try:
            cur = stream
            for stage, meta in reversed(dec_meta):
                cur = stage["dec"](cur, meta)
            return cur == target
        except Exception:
            return False

    def walk(state, used):
        seq = state["seq"]
        names = state["stages"]
        dec_meta = state["dec_meta"]

        if not names:
            add("UTF-8 brut", raw_bits, True, "référence", names, None)
        else:
            ok = state["ok"]
            if len(names) == 1:
                label = "RLE classique (solo)" if names[0] == "RLE" else f"{names[0]} (solo)"
                add(label, state["plain_bits"], ok, "solo", names, None)
            else:
                add(f"{'+'.join(names)} (sans entropie)",
                    state["plain_bits"], ok, "combiné", names, None)

        for C in (Huffman, Arith):
            label = f"{'+'.join(names)}+{C.label}" if names else C.label
            category = "entropie" if not names else "combiné"
            try:
                b = C.enc(seq)
                ok = verify(C.dec(b), dec_meta)
                add(label, b["payload"], ok, category, names, C.label)
            except Exception:
                add(label, failure_payload, False, category, names, C.label)

        if len(names) >= max_stages:
            return
        for stage in stage_list:
            if stage["name"] in used:
                continue
            try:
                out, meta, plain = stage["enc"](seq)
                step_ok = stage["dec"](out, meta) == seq
            except Exception:
                continue
            walk({
                "seq": out,
                "stages": names + [stage["name"]],
                "dec_meta": dec_meta + [(stage, meta)],
                "plain_bits": plain,
                "ok": state["ok"] and step_ok,
            }, used | {stage["name"]})

    walk({"seq": target, "stages": [],
          "dec_meta": [], "plain_bits": raw_bits, "ok": True}, frozenset())
    return results


def parse_stages_arg(spec):
    """Sélectionne les préprocesseurs autorisés, dans l'ordre canonique."""
    if not spec:
        return None
    names = [part.strip().upper() for part in spec.split(",") if part.strip()]
    if not names:
        raise ValueError("liste de préprocesseurs vide")
    unknown = [name for name in names if name not in STAGE_BY_NAME]
    if unknown:
        raise ValueError(f"technique(s) inconnue(s) : {', '.join(unknown)}")
    return [stage for stage in STAGES if stage["name"] in names]


def parse_pipeline_arg(spec):
    """Valide une chaîne unique, par exemple RLE+LZW+Huffman."""
    tokens = [part.strip() for part in re.split(r"[+,]", spec) if part.strip()]
    if not tokens:
        raise ValueError("chaîne de pipeline vide")
    chain, seen, coder = [], set(), None
    for position, token in enumerate(tokens):
        key = token.lower()
        stage_name = next((name for name in STAGE_BY_NAME
                           if name.lower() == key), None)
        if stage_name is not None:
            if coder is not None:
                raise ValueError("le codeur doit être en fin de chaîne")
            if stage_name in seen:
                raise ValueError(f"{stage_name} apparaît deux fois")
            seen.add(stage_name)
            chain.append(stage_name)
            continue
        aliases = {
            Huffman.label.lower(): Huffman.label,
            "arith": Arith.label,
            "arithmetique": Arith.label,
            "arithmétique": Arith.label,
        }
        coder_name = aliases.get(key)
        if coder_name is None:
            raise ValueError(f"technique inconnue : {token}")
        if coder is not None:
            raise ValueError("un seul codeur terminal autorisé")
        if position != len(tokens) - 1:
            raise ValueError("le codeur doit être en dernier")
        coder = coder_name
        chain.append(coder_name)
    return chain


def run_selected_pipeline(text: str, chain, raw_bits: int) -> Dict[str, Any]:
    """Évalue une seule chaîne, sans parcourir toutes les combinaisons."""
    target = list(text)
    seq, names, dec_meta = target, [], []
    plain, coder_name = raw_bits, None

    for token in chain:
        stage = STAGE_BY_NAME.get(token)
        if stage is None:
            coder_name = token
            break
        out, meta, bits = stage["enc"](seq)
        names.append(token)
        dec_meta.append((stage, meta))
        seq, plain = out, bits

    coder = CODER_MAP.get(coder_name)
    if coder is not None:
        b = coder.enc(seq)
        try:
            coder_ok = coder.dec(b) == seq
        except Exception:
            coder_ok = False
        cur = coder.dec(b)
        for stage, meta in reversed(dec_meta):
            cur = stage["dec"](cur, meta)
        payload = b["payload"]
        ok = coder_ok and cur == target
    else:
        cur = seq
        for stage, meta in reversed(dec_meta):
            cur = stage["dec"](cur, meta)
        payload = plain
        ok = cur == target

    if coder_name is not None:
        category = "entropie" if not names else "combiné"
    elif len(names) == 1:
        category = "solo"
    else:
        category = "combiné"
    return {
        "label": "+".join(chain),
        "payload": payload,
        "ok": ok,
        "category": category,
        "stages": names,
        "coder": coder_name,
    }


def short_value(value, limit=64):
    text = repr(value)
    return text if len(text) <= limit else text[:limit - 1] + "…"


LOG_ROWS = 500


def _chain(rec):
    chain = list(rec.get("stages", []))
    if rec.get("coder"):
        chain.append(rec["coder"])
    return chain


def _stage_detail(stage):
    """Détail vérifiable d'une étape : (en-têtes, lignes, alignements)."""
    name, out = stage["name"], stage["output"]
    if name == "LZ77":
        return (
            ["#", "Position", "Type", "Distance", "Longueur", "Symbole suivant", "Bits"],
            lz77_token_rows(out),
            ["right", "right", "center", "right", "right", "left", "right"],
        )
    if name == "LZ78":
        return (
            ["#", "Position", "Index dict.", "Symbole", "Longueur", "Bits"],
            lz78_token_rows(out),
            ["right", "right", "right", "left", "right", "right"],
        )
    if name == "LZW":
        return (
            ["#", "Indexe", "Dictionnaire", "Encodage"],
            lzw_token_rows(stage["input"]),
            ["right", "right", "left", "right"],
        )
    if name == "RLE":
        return (
            ["#", "Symbole", "Longueur", "Bits"],
            rle_token_rows(out),
            ["right", "left", "right", "right"],
        )
    if name == "BWT":
        rows = [[str(i), short_value(sym, 40), str(sym_bits(sym))]
                for i, sym in enumerate(out)]
        return (
            ["Position", "Dernière colonne", "Bits"],
            rows,
            ["right", "left", "right"],
        )
    rows = [[str(i + 1), short_value(sym, 40)] for i, sym in enumerate(out)]
    return ["#", "Symbole"], rows, ["right", "left"]


def arith_final_state(stream, b):
    """Intervalle final [Ln/den, (Ln+Rn)/den[ du codage arithmétique."""
    n = b["N"]
    counts = dict(zip(b["syms"], b["counts"]))
    cum, acc = {}, 0
    for sym, count in zip(b["syms"], b["counts"]):
        cum[sym] = acc
        acc += count
    left, right, den = 0, 1, 1
    for sym in stream:
        left = left * n + right * cum[sym]
        right *= counts[sym]
        den *= n
    return left, right, den


def arith_fit_ok(b, left, right, den):
    """Vérifie que le code m/2^k tombe bien dans l'intervalle final ouvert."""
    k, m = b["k"], b["m"]
    return (left << k) < m * den < ((left + right) << k)


def long_int(value, limit=800):
    try:
        text = str(value)
    except ValueError:
        return f"<{value.bit_length()} bits>"
    return text if len(text) <= limit else text[:limit] + f"… ({len(text)} chiffres)"


def seq_preview(seq, limit=120, width=600):
    if not seq:
        return "∅"
    body = repr(list(seq[:limit]))
    if len(seq) > limit:
        body += f" ... ({len(seq)} symboles au total)"
    return body if len(body) <= width else body[:width - 1] + "…"


def seq_to_text(seq):
    try:
        return "".join(seq)
    except TypeError:
        return "".join(str(s) for s in seq)


def explain_pipeline(text, rec, raw_bits, decimals=4, trace=None):
    """Démarche complète et vérifiable du meilleur pipeline."""
    if trace is None:
        trace = build_trace(text, rec, raw_bits)

    animate_print(f"[+] DÉMARCHE DE CALCUL : {rec['label']}", "32", 0.004)
    print(tag("*", f"Entrée : {len(text)} caractères → "
                   f"{len(text.encode('utf-8'))} octets ({raw_bits} bits)"))
    chain = _chain(rec)
    if chain:
        animate_print("  " + "  ──▶  ".join(chain), "36", 0.003)
    print(tag("*", "Chaque étape est recalculée puis décodée pour vérifier le résultat."))

    total_steps = len(trace["stages"]) + (1 if trace["coder"] is not None else 0)
    for index, stage in enumerate(trace["stages"], 1):
        summary = stage_summary_rows(stage)
        render_table(
            f"ÉTAPE {index}/{total_steps} · {stage['name']} — RÉSUMÉ DU CALCUL",
            ["Grandeur", "Valeur", "Formule / contrôle"],
            summary,
            ["left", "right", "left"],
        )
    coder = trace["coder"]
    if coder is not None:
        b = coder["b"]
        if coder["name"] == "Huffman":
            avg = trunc_dec(b["payload"], max(1, len(coder["input"])), 2)
            summary = [
                ["Symboles à coder", str(len(coder["input"])), "longueur du flux d'entrée"],
                ["Symboles distincts", str(len(b["table"])), "taille de la table"],
                ["Payload", f"{b['payload']} bits", "Σ(fréquence × longueur)"],
                ["Longueur moyenne", f"{avg} bits/symbole", "payload / nombre de symboles"],
                ["Décodage", "OK" if coder["ok"] else "ÉCHEC", "encode → decode"],
            ]
            step = len(trace["stages"]) + 1
            render_table(
                f"ÉTAPE {step}/{total_steps} · HUFFMAN — CODEUR TERMINAL",
                ["Grandeur", "Valeur", "Formule / contrôle"],
                summary, ["left", "right", "left"]
            )
        else:
            left, right, den = arith_final_state(coder["input"], b)
            fit = arith_fit_ok(b, left, right, den)
            avg = trunc_dec(b["payload"], max(1, b["N"]), 2)
            summary = [
                ["N", str(b["N"]), "nombre de symboles"],
                ["Alphabet", str(len(b["syms"])), "symboles distincts"],
                ["k", f"{b['k']} bits", "taille du code émis"],
                ["m", short_int(b["m"]), "valeur entière émise"],
                ["Payload", f"{b['payload']} bits", "payload = k"],
                ["Longueur moyenne", f"{avg} bits/symbole", "k / N"],
                ["Intervalle final", "OK" if fit else "ÉCHEC",
                 "Ln/den ≤ m/2^k < (Ln+Rn)/den"],
                ["Décodage", "OK" if coder["ok"] else "ÉCHEC", "encode → decode"],
            ]
            step = len(trace["stages"]) + 1
            render_table(
                f"ÉTAPE {step}/{total_steps} · ARITHMÉTIQUE — CODEUR TERMINAL",
                ["Grandeur", "Valeur", "Formule / contrôle"],
                summary, ["left", "right", "left"]
            )
    steps_ok = all(s["ok"] for s in trace["stages"]) and (coder is None or coder["ok"])
    final_size = trace["final_payload"]
    verification = trace["final_ok"] and steps_ok
    coherent = (trace["final_payload"] == rec["payload"]
                and verification == rec["ok"])

    final_rows = [
        ["Taille originale", f"{raw_bits} bits",
         f"{len(text.encode('utf-8'))} octets × 8"],
        ["Payload final", f"{trace['final_payload']} bits",
         "sortie du dernier étage / codeur"],
        ["Taille retenue", f"{final_size} bits", "payload seul"],
        ["Taux de réduction", f"{gain_str(raw_bits, final_size, decimals)} %",
         "(original - retenue) / original × 100"],
        ["Débit", f"{trunc_dec(trace['final_payload'], max(1, len(text)), 2)} bits/car.",
         "payload / nombre de caractères"],
        ["Vérification", "OK" if verification else "ÉCHEC",
         "encode → decode (toutes les étapes)"],
        ["Cohérence", "OK" if coherent else "ÉCHEC",
         "recalcul = tableau comparatif"],
    ]
    render_table(
        "CALCUL FINAL — TAILLE ET TAUX DE RÉDUCTION",
        ["Élément", "Valeur", "Formule / contrôle"],
        final_rows, ["left", "right", "left"]
    )

    if coherent and verification:
        print(tag("+", "Contrôle de cohérence : OK — le détail reproduit exactement "
                       "le résultat affiché."))
    elif not coherent:
        print(tag("!", "Contrôle incohérent entre le recalcul et le tableau."))
    if not verification:
        print(tag("x", "ÉCHEC de la vérification sans perte."))


def write_verification_log(path, text, rec, trace, raw_bits, decimals=4):
    """Écrit un journal texte complet, indépendant de l'affichage console."""
    if trace is None:
        trace = build_trace(text, rec, raw_bits)

    sep = "=" * 78
    sub = "-" * 78
    lines = []
    add = lines.append

    chain = " -> ".join(_chain(rec)) or "UTF-8 brut"
    final_size = trace["final_payload"]
    steps_ok = (all(s["ok"] for s in trace["stages"])
                and (trace["coder"] is None or trace["coder"]["ok"]))
    verification = trace["final_ok"] and steps_ok
    coherent = (trace["final_payload"] == rec["payload"]
                and verification == rec["ok"])
    decoded = seq_to_text(trace["decoded"])
    src_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    dec_sha = hashlib.sha256(decoded.encode("utf-8")).hexdigest()
    total_steps = len(trace["stages"]) + (1 if trace["coder"] is not None else 0)

    add(sep)
    add("COMPRESS-LAB · JOURNAL DE VÉRIFICATION DU MEILLEUR PIPELINE")
    add(sep)
    add(f"Date                : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"Commande            : {' '.join(sys.argv)}")
    add(f"Pipeline            : {rec['label']}")
    add(f"Chaîne              : {chain}")
    add(f"Mesure              : payload uniquement")
    add(f"Source              : {len(text)} caractères -> "
        f"{len(text.encode('utf-8'))} octets ({raw_bits} bits)")
    add(f"SHA-256 source      : {src_sha}")
    add(f"Aperçu source       : {short_value(text, 200)}")
    add(f"Résultat affiché    : payload={rec['payload']} bits, "
        f"retenu={final_size} bits, "
        f"réduction={gain_str(raw_bits, final_size, decimals)} %")
    add("")

    if trace["stages"]:
        add(sub)
        add("CALCUL DÉTAILLÉ DES ÉTAPES")
        add(sub)

    for index, stage in enumerate(trace["stages"], 1):
        name, out = stage["name"], stage["output"]
        add("")
        add(f"ÉTAPE {index}/{total_steps} · {name}")
        add(sub)
        add(f"Entrée              : {len(stage['input'])} symboles")
        if name == "LZ77":
            n_lit = sum(1 for tok in out if tok[0] == 'L')
            add(f"Sortie              : {len(out)} tokens "
                f"({n_lit} littéraux, {len(out) - n_lit} copies)")
        elif name == "LZ78":
            add(f"Sortie              : {len(out)} tokens")
        elif name == "LZW":
            add(f"Sortie              : {len(out)} codes")
            add(f"Alphabet            : {len(stage['meta']['alphabet'])} symboles")
        elif name == "RLE":
            add(f"Sortie              : {len(out) // 2} runs")
        else:
            add(f"Sortie              : {len(out)} symboles")
        add(f"Payload calculé     : {stage['plain_bits']} bits")
        add(f"Décodage de l'étape : {'OK' if stage['ok'] else 'ÉCHEC'}")
        if name == "BWT":
            add(f"Index primaire      : {stage['meta']['idx']}")
            add(f"Alphabet            : {len(stage['meta']['alpha'])} symboles")
        add(f"Aperçu entrée       : {seq_preview(stage['input'])}")
        add(f"Aperçu sortie       : {seq_preview(out)}")
        add("")
        add("Résumé du calcul :")
        lines.extend(plain_table_lines(
            ["Grandeur", "Valeur", "Formule / contrôle"],
            stage_summary_rows(stage),
            ["left", "right", "left"]
        ))
        headers, rows, aligns = _stage_detail(stage)
        if rows:
            add("")
            add(f"Détail ({len(rows)} lignes) :")
            if len(rows) > LOG_ROWS:
                add(f"Affichage des {LOG_ROWS} premières lignes sur {len(rows)}.")
                rows = rows[:LOG_ROWS]
            lines.extend(plain_table_lines(headers, rows, aligns))

    coder = trace["coder"]
    if coder is not None:
        b = coder["b"]
        add("")
        add(sub)
        add(f"ÉTAPE {len(trace['stages']) + 1}/{total_steps} · "
            f"{coder['name'].upper()} — CODEUR TERMINAL")
        add(sub)
        add(f"Entrée              : {len(coder['input'])} symboles")
        add(f"Décodage du codeur  : {'OK' if coder['ok'] else 'ÉCHEC'}")
        if coder["name"] == "Huffman":
            add(f"Payload             : {b['payload']} bits")
            add("")
            rows = huffman_detail_rows(coder["input"], b)
            add(f"Table des codes ({len(rows)} symboles) :")
            lines.extend(plain_table_lines(
                ["Symbole", "Fréquence", "Longueur", "Code", "Fréq. × longueur"],
                rows,
                ["left", "right", "right", "center", "right"]
            ))
            add(f"Somme fréquence × longueur = "
                f"{sum(int(r[4]) for r in rows)} bits "
                f"(payload = {b['payload']} bits)")
        else:
            add(f"N                   : {b['N']}")
            add(f"Alphabet            : {len(b['syms'])} symboles")
            add(f"k                   : {b['k']} bits")
            add(f"m                   : {long_int(b['m'])}")
            add(f"Payload             : {b['payload']} bits")
            left, right, den = arith_final_state(coder["input"], b)
            add(f"Intervalle final Ln : {long_int(left)}")
            add(f"Intervalle final Rn : {long_int(right)}")
            add(f"Dénominateur den    : {long_int(den)}")
            add(f"Contrôle intervalle : "
                f"{'OK' if arith_fit_ok(b, left, right, den) else 'ÉCHEC'} "
                f"(Ln/den < m/2^k < (Ln+Rn)/den)")
            add("")
            freq_rows = arith_freq_rows(b)
            add(f"Table de fréquences ({len(freq_rows)} symboles) :")
            if len(freq_rows) > LOG_ROWS:
                add(f"Affichage des {LOG_ROWS} premières lignes sur {len(freq_rows)}.")
                freq_rows = freq_rows[:LOG_ROWS]
            lines.extend(plain_table_lines(
                ["i", "s_i", "n_i", "P(s_i)", "C(s_i-1)", "C(s_i)"],
                freq_rows,
                ["right", "left", "right", "right", "right", "right"]
            ))
            add("")
            add("Réduction de l'intervalle :")
            interval_rows = arith_interval_rows(
                coder["input"], b, min(b["N"], 120)
            )
            lines.extend(plain_table_lines(
                ["j", "i", "donnée", "L_j", "R_j"],
                interval_rows,
                ["right", "right", "left", "right", "right"]
            ))
            if b["N"] > 120:
                add(f"({120} premières étapes sur {b['N']})")

    add("")
    add(sub)
    add("CALCUL FINAL")
    add(sub)
    lines.extend(plain_table_lines(
        ["Élément", "Valeur", "Formule / contrôle"],
        [
            ["Taille originale", f"{raw_bits} bits",
             f"{len(text.encode('utf-8'))} octets x 8"],
            ["Payload final", f"{trace['final_payload']} bits",
             "sortie du dernier étage / codeur"],
            ["Taille retenue", f"{final_size} bits", "payload seul"],
            ["Taux de réduction", f"{gain_str(raw_bits, final_size, decimals)} %",
             "(original - retenue) / original x 100"],
            ["Vérification", "OK" if verification else "ÉCHEC",
             "encode -> decode"],
            ["Cohérence", "OK" if coherent else "ÉCHEC",
             "recalcul = tableau comparatif"],
        ],
        ["left", "right", "left"]
    ))
    add("")
    add(f"Recalcul            : payload={trace['final_payload']} bits, "
        f"retenu={final_size} bits")
    add(f"Tableau comparatif  : payload={rec['payload']} bits, "
        f"total={rec['payload']} bits")
    add(f"Écart               : payload={trace['final_payload'] - rec['payload']} bits")
    add(f"SHA-256 source      : {src_sha}")
    add(f"SHA-256 décodé      : {dec_sha}")
    add(f"Texte décodé        : {short_value(decoded, 200)}")
    add(f"Vérification        : "
        f"{'OK — encode -> decode identique au texte source' if verification else 'ÉCHEC'}")
    add(f"RÉSULTAT            : {'COHÉRENT' if coherent and verification else 'INCOHÉRENT'}")
    add(sep)

    try:
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(lines) + "\n")
    except OSError as exc:
        print(tag("x", f"Impossible d'écrire le journal {path} : {exc}"),
              file=sys.stderr)
        return None
    return path


def short_int(value, limit=30):
    try:
        text = str(value)
    except ValueError:
        return f"<{value.bit_length()} bits>"
    return text if len(text) <= limit else text[:12] + "…" + text[-10:]


def plain_table_lines(headers, rows, aligns):
    real = [r for r in rows if r is not None]
    widths = [max([len(headers[i])] + [len(str(r[i])) for r in real])
              for i in range(len(headers))]

    def bar(left, mid, right):
        return left + mid.join("-" * (w + 2) for w in widths) + right

    def pad(value, index):
        text = str(value)
        if aligns[index] == "right":
            return text.rjust(widths[index])
        if aligns[index] == "center":
            return text.center(widths[index])
        return text.ljust(widths[index])

    lines = [bar("+", "+", "+")]
    lines.append("| " + " | ".join(headers[i].center(widths[i])
                                   for i in range(len(headers))) + " |")
    lines.append(bar("+", "+", "+"))
    for row in rows:
        if row is None:
            lines.append("| " + " | ".join("...".center(w) for w in widths) + " |")
        else:
            lines.append("| " + " | ".join(pad(v, i) for i, v in enumerate(row)) + " |")
    lines.append(bar("+", "+", "+"))
    return lines


def lz77_token_rows(tokens):
    rows, pos = [], 0
    for number, tok in enumerate(tokens, 1):
        if tok[0] == 'L':
            bits = 1 + sym_bits(tok[1])
            rows.append([str(number), str(pos), "littéral", "-", "-",
                         short_value(tok[1], 32), str(bits)])
            pos += 1
        else:
            dist, length, nxt = tok[1], tok[2], tok[3]
            bits = 1 + bits_for(dist) + bits_for(length)
            if nxt is not NO_SYMBOL:
                bits += sym_bits(nxt)
            rows.append([str(number), str(pos), "copie", str(dist), str(length),
                         short_value(nxt, 32), str(bits)])
            pos += length + (0 if nxt is NO_SYMBOL else 1)
    return rows


def lz78_token_rows(tokens):
    rows, dictionary, pos = [], {0: ()}, 0
    for number, tok in enumerate(tokens, 1):
        idx, sym = tok[1], tok[2]
        entry = dictionary[idx] + (() if sym is NO_SYMBOL else (sym,))
        bits = bits_for(idx) + (1 if sym is NO_SYMBOL else sym_bits(sym))
        rows.append([str(number), str(pos), str(idx), short_value(sym, 32),
                     str(len(entry)), str(bits)])
        dictionary[number] = entry
        pos += len(entry)
    return rows


def lzw_encode_steps(seq):
    """Étapes d'encodage LZW, pour l'affichage du détail."""
    alphabet = lzw_alphabet(seq)
    dictionary = {(sym,): idx for idx, sym in enumerate(alphabet, 1)}
    next_idx = len(alphabet) + 1
    steps = [{"index": idx, "entry": (sym,), "code": None}
             for idx, sym in enumerate(alphabet, 1)]
    codes, w = [], ()
    for sym in seq:
        wc = w + (sym,)
        if wc in dictionary:
            w = wc
        else:
            code = dictionary[w]
            codes.append(code)
            dictionary[wc] = next_idx
            steps.append({"index": next_idx, "entry": wc, "code": code})
            next_idx += 1
            w = (sym,)
    if w:
        code = dictionary[w]
        codes.append(code)
        steps.append({"index": None, "entry": None, "code": code})
    return codes, steps


def lzw_entry_text(entry):
    if all(isinstance(sym, str) and len(sym) == 1 for sym in entry):
        return "".join(entry)
    return "·".join(short_value(sym, 24) for sym in entry)


def lzw_token_rows(seq):
    _, steps = lzw_encode_steps(seq)
    rows = []
    for number, step in enumerate(steps, 1):
        index = "—" if step["index"] is None else str(step["index"])
        entry = "—" if step["entry"] is None else short_value(
            lzw_entry_text(step["entry"]), 40)
        code = "—" if step["code"] is None else str(step["code"])
        rows.append([str(number), index, entry, code])
    return rows


def rle_token_rows(stream):
    rows = []
    for i in range(0, len(stream), 2):
        sym, count = stream[i], stream[i + 1]
        bits = sym_bits(sym) + bits_for(count)
        rows.append([str(i // 2 + 1), short_value(sym, 32), str(count), str(bits)])
    return rows


def huffman_detail_rows(stream, b):
    counts = Counter(stream)
    codes = canonical(b["table"])
    rows = []
    for sym, length in b["table"]:
        freq = counts[sym]
        rows.append([short_value(sym, 36), str(freq), str(length),
                     codes[sym], str(freq * length)])
    return rows


def arith_dec4(num, den, digits=4):
    """Valeur num/den tronquée à 4 chiffres après la virgule."""
    scale = 10 ** digits
    q = (num * scale) // den
    return f"{q // scale},{q % scale:0{digits}d}"


def arith_freq_rows(b):
    rows, cum = [], 0
    total = b["N"]
    for i, (sym, count) in enumerate(zip(b["syms"], b["counts"]), 1):
        rows.append([
            str(i),
            short_value(sym, 36),
            str(count),
            arith_dec4(count, total),
            arith_dec4(cum, total),
            arith_dec4(cum + count, total),
        ])
        cum += count
    return rows


def arith_interval_rows(stream, b, limit=10):
    cum, acc = {}, 0
    for sym, count in zip(b["syms"], b["counts"]):
        cum[sym] = acc
        acc += count
    counts = dict(zip(b["syms"], b["counts"]))
    index = {sym: i for i, sym in enumerate(b["syms"], 1)}
    n = b["N"]
    left, right, den = 0, 1, 1
    rows = [["0", "—", "départ", arith_dec4(0, 1), arith_dec4(1, 1)]]
    for number, sym in enumerate(stream[:limit], 1):
        left = left * n + right * cum[sym]
        right *= counts[sym]
        den *= n
        rows.append([
            str(number),
            str(index[sym]),
            short_value(sym, 28),
            arith_dec4(left, den),
            arith_dec4(left + right, den),
        ])
    return rows


def stage_summary_rows(stage):
    name, out = stage["name"], stage["output"]
    plain = stage["plain_bits"]
    if name == "LZ77":
        n_lit = sum(1 for tok in out if tok[0] == 'L')
        n_match = len(out) - n_lit
        lit_bits = sum(1 + sym_bits(tok[1]) for tok in out if tok[0] == 'L')
        match_bits = plain - lit_bits
        return [
            ["Littéraux", str(n_lit), f"Σ(1 + bits(symbole)) = {lit_bits} bits"],
            ["Copies", str(n_match), f"Σ(1 + bits(distance) + bits(longueur) + bits(symbole)) = {match_bits} bits"],
            ["Payload LZ77", f"{plain} bits", f"{lit_bits} + {match_bits} = {plain} bits"],
        ]
    if name == "LZ78":
        max_idx = max((tok[1] for tok in out), default=0)
        idx_bits = sum(bits_for(tok[1]) for tok in out)
        sym_bits_sum = sum(1 if tok[2] is NO_SYMBOL else sym_bits(tok[2])
                           for tok in out)
        table_bits = bits_for(max_idx) * 2
        return [
            ["Tokens", str(len(out)), "un token = (index, symbole)"],
            ["Bits index", f"{idx_bits} bits", "Σ bits(index)"],
            ["Bits symboles", f"{sym_bits_sum} bits", "Σ bits(symbole)"],
            ["Bits table", f"{table_bits} bits", "2 × bits(index max)"],
            ["Payload LZ78", f"{plain} bits", f"{idx_bits} + {sym_bits_sum} + {table_bits} = {plain} bits"],
        ]
    if name == "LZW":
        alphabet = stage["meta"]["alphabet"]
        width = bits_for(max(out, default=0))
        code_bits = len(out) * width
        alpha_bits = sum(sym_bits(sym) for sym in alphabet)
        return [
            ["Alphabet", str(len(alphabet)), "symboles distincts"],
            ["Codes", str(len(out)), "un code par entrée émise"],
            ["Bits par code", f"{width} bits", "bits(index max)"],
            ["Bits des codes", f"{code_bits} bits", f"{len(out)} × {width}"],
            ["Bits alphabet", f"{alpha_bits} bits", "Σ bits(symbole)"],
            ["Payload LZW", f"{plain} bits", f"{code_bits} + {alpha_bits} = {plain} bits"],
        ]
    if name == "RLE":
        runs = len(out) // 2
        sym_bits_sum = sum(sym_bits(out[i]) for i in range(0, len(out), 2))
        len_bits_sum = sum(bits_for(out[i]) for i in range(1, len(out), 2))
        return [
            ["Runs", str(runs), "couples (symbole, longueur)"],
            ["Bits symboles", f"{sym_bits_sum} bits", "Σ bits(symbole)"],
            ["Bits longueurs", f"{len_bits_sum} bits", "Σ bits(longueur)"],
            ["Payload RLE", f"{plain} bits", f"{sym_bits_sum} + {len_bits_sum} = {plain} bits"],
        ]
    if name == "BWT":
        return [
            ["n", str(len(out)), "taille de la dernière colonne"],
            ["Index primaire", str(stage["meta"]["idx"]), "position de la rotation originale"],
            ["Alphabet", str(len(stage["meta"]["alpha"])), "symboles distincts"],
            ["Payload BWT", f"{plain} bits", "Σ bits(symboles de la dernière colonne)"],
        ]
    return [["Payload", f"{plain} bits", "sortie estimée"]]


def build_trace(text, rec, raw_bits):
    target = list(text)
    stage_map = STAGE_BY_NAME
    coder_map = CODER_MAP
    seq = target
    current_bits = raw_bits
    dec_meta = []
    stages = []

    for name in rec.get("stages", []):
        stage = stage_map[name]
        in_seq = seq
        out, meta, plain = stage["enc"](seq)
        try:
            ok = stage["dec"](out, meta) == in_seq
        except Exception:
            ok = False
        stages.append({
            "name": name, "input": in_seq, "output": out,
            "meta": meta, "plain_bits": plain, "ok": ok,
        })
        seq = out
        current_bits = plain
        dec_meta.append((stage, meta))

    coder = coder_map.get(rec.get("coder"))
    coder_trace = None
    if coder is not None:
        b = coder.enc(seq)
        try:
            coder_ok = coder.dec(b) == seq
        except Exception:
            coder_ok = False
        coder_trace = {"name": rec["coder"], "input": seq, "b": b, "ok": coder_ok}
        cur = coder.dec(b)
        for stage, meta in reversed(dec_meta):
            cur = stage["dec"](cur, meta)
        final_ok = cur == target
        final_payload = b["payload"]
    else:
        cur = seq
        for stage, meta in reversed(dec_meta):
            cur = stage["dec"](cur, meta)
        final_ok = cur == target
        final_payload = current_bits

    return {
        "target": target, "raw_bits": raw_bits, "stages": stages,
        "coder": coder_trace, "final_payload": final_payload,
        "final_ok": final_ok, "decoded": cur,
    }


def log_path_for(label):
    slug = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "_", slug.lower()).strip("_") or "best"
    return f"verification_{slug}.log"


def load_text(args) -> str:
    if args.file:
        path = args.file
        if not os.path.isfile(path):
            print(tag("x", f"Fichier introuvable : {path}"), file=sys.stderr)
            sys.exit(1)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    if args.text is not None:
        if os.path.isfile(args.text):
            with open(args.text, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        return args.text

    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if data.strip():
            return data

    return DEFAULT_TEXT

def main():
    ap = argparse.ArgumentParser(
        description="Banc d'essai de compression (BWT/RLE/Huffman/Arith/LZ77/LZ78/LZW)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python3 compress_lab.py
  python3 compress_lab.py "mon texte"
  python3 compress_lab.py -f monfichier.txt
  python3 compress_lab.py monfichier.txt
  cat monfichier.txt | python3 compress_lab.py
  python3 compress_lab.py --solo-only
  python3 compress_lab.py --combined-only
  python3 compress_lab.py --max-stages 2
  python3 compress_lab.py --stages RLE,LZW --max-stages 2
  python3 compress_lab.py --pipeline "RLE+LZW+Huffman"
  python3 compress_lab.py --quick
  python3 compress_lab.py --no-explain
  python3 compress_lab.py --log mon_log.txt
  python3 compress_lab.py --no-log
  python3 compress_lab.py --no-color -d 2
"""
    )
    ap.add_argument("text", nargs="?", default=None, help="Texte ou chemin de fichier")
    ap.add_argument("-f", "--file", metavar="PATH", help="Lire depuis un fichier")
    ap.add_argument("-d", "--decimals", type=int, default=4)
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--solo-only", action="store_true", help="Uniquement les techniques pures")
    ap.add_argument("--combined-only", action="store_true", help="Uniquement les pipelines combinés")
    ap.add_argument("--stages", metavar="LISTE",
                    help="Préprocesseurs à combiner, séparés par des virgules "
                         "(ex. RLE,LZW ; défaut : tous)")
    ap.add_argument("--pipeline", metavar="CHAÎNE",
                    help="Tester une seule chaîne (ex. RLE+LZW+Huffman)")
    ap.add_argument("--max-stages", type=int, default=None, choices=(1, 2, 3, 4, 5),
                    help="Nombre max de préprocesseurs enchaînés "
                         "(défaut : 4 ; 2 en mode --quick ; 5 = tous)")
    ap.add_argument("--top", type=int, default=None, metavar="N",
                    help="N'afficher que les N meilleurs pipelines")
    ap.add_argument("--show-all", action="store_true",
                    help="Afficher aussi les pipelines à gain négatif")
    ap.add_argument("--quick", action="store_true",
                    help="Mode rapide : 2 étages max, top 15, sans détail ni journal")
    ap.add_argument("--no-explain", action="store_true",
                    help="Ne pas afficher la démarche détaillée du meilleur pipeline")
    ap.add_argument("--no-animation", action="store_true",
                    help="Désactiver l'animation colorée de la démarche")
    ap.add_argument("--log", metavar="PATH", default=None,
                    help="Chemin du journal de vérification "
                         "(défaut : verification_<pipeline>.log)")
    ap.add_argument("--no-log", action="store_true",
                    help="Ne pas écrire le journal de vérification")
    args = ap.parse_args()

    try:
        selected_stages = parse_stages_arg(args.stages)
        selected_pipeline = (parse_pipeline_arg(args.pipeline)
                             if args.pipeline else None)
    except ValueError as exc:
        ap.error(str(exc))

    if args.quick:
        args.no_explain = True
        args.no_log = True
        args.no_animation = True
        if args.top is None:
            args.top = 15
    if args.max_stages is None:
        args.max_stages = 2 if args.quick else 4
    if args.top is not None and args.top < 1:
        ap.error("--top doit être un entier ≥ 1")

    Ansi.enabled = sys.stdout.isatty() and not args.no_color
    Animate.enabled = Ansi.enabled and not args.no_animation

    text = load_text(args).rstrip("\n")
    if not text:
        print(tag("x", "Texte vide – rien à compresser."), file=sys.stderr)
        sys.exit(1)

    raw_bytes = len(text.encode("utf-8"))
    raw_bits = raw_bytes * 8

    print(paint(BANNER, "1", "32"))
    src = "fichier" if (args.file or (args.text and os.path.isfile(str(args.text)))) else "argument/stdin/défaut"
    print(tag("*", f"Source : {src}"))
    print(tag("*", f"Texte original : {len(text)} caractères  →  {raw_bytes} octets  ({raw_bits} bits)"))
    active = [stage["name"] for stage in (selected_stages or STAGES)]
    print(tag("*", f"Préprocesseurs : {', '.join(active)} · "
                   f"codeurs : Huffman, Arithmétique"))
    print(tag("*", "Tous les pipelines sont vérifiés sans perte (encode → decode)"))
    print(tag("*", "Mode de calcul : payload uniquement"))

    if selected_pipeline is not None:
        results = [run_selected_pipeline(text, selected_pipeline, raw_bits)]
    else:
        max_stages = 1 if args.solo_only else args.max_stages
        results = run_all_pipelines(text, max_stages=max_stages,
                                    stages=selected_stages)
        total_combos = len(results)
        print(tag("*", f"Pipelines évalués : {total_combos} "
                       f"(séquences ordonnées, Huffman/Arithmétique en terminal)"))

        if args.solo_only:
            results = [r for r in results
                       if r["category"] in ("solo", "entropie", "référence")]
        elif args.combined_only:
            results = [r for r in results
                       if r["category"] in ("combiné", "entropie", "référence")]

    def size_of(r):
        return r["payload"]

    if selected_pipeline is not None:
        tested = results[0]
        print(tag("*", f"Pipeline demandé : {tested['label']} "
                       f"({tested['payload']} bits, "
                       f"{'OK' if tested['ok'] else 'ÉCHEC'})"))
    elif args.show_all:
        print(tag("*", f"Pipelines affichés : {len(results)} "
                       f"(tous, gain < 0 inclus)"))
    else:
        candidates = len(results)
        results = [r for r in results if size_of(r) <= raw_bits]
        hidden = candidates - len(results)
        print(tag("*", f"Pipelines affichés : {len(results)} (gain ≥ 0)"
                       + (f" · {hidden} masqué(s) (gain < 0)" if hidden else "")))

    if not results:
        print(tag("!", "Aucun pipeline avec un gain positif ou nul."))
        return

    results.sort(key=size_of)
    evaluated = len(results)
    if args.top is not None and len(results) > args.top:
        results = results[:args.top]
        print(tag("*", f"Affichage limité aux {len(results)} meilleurs "
                       f"sur {evaluated}."))

    rows = []
    for rank, r in enumerate(results, 1):
        size = size_of(r)
        rows.append([
            str(rank), r["label"], str(r["payload"]),
            trunc_dec(size, 8, args.decimals),
            gain_str(raw_bits, size, args.decimals) + " %",
            "OK" if r["ok"] else "ÉCHEC",
        ])
    headers = ["#", "Pipeline", "Payload (bits)", "Octets", "Gain", "Sans perte"]
    aligns = ["right", "left", "right", "right", "right", "center"]

    scope = ("tous, gain < 0 inclus"
             if selected_pipeline is not None or args.show_all else "gain ≥ 0")
    render_table(
        f"COMPARATIF DES PIPELINES (payload seul, {scope}, triés par taille)",
        headers, rows, aligns
    )

    best = results[0]
    best_size = size_of(best)
    print()
    prefix = "Pipeline testé" if selected_pipeline is not None else "Meilleur pipeline"
    print(tag("+", f"{prefix} : {best['label']}"))
    print(tag("+", f"Taille retenue (payload seul) : {best_size} bits ≈ "
                   f"{trunc_dec(best_size, 8, 2)} octets"))
    print(tag("+", f"Gain de compression : {gain_str(raw_bits, best_size, 2)} %"))

    trace = None
    if not args.no_explain or not args.no_log:
        trace = build_trace(text, best, raw_bits)

    if not args.no_explain:
        explain_pipeline(text, best, raw_bits, args.decimals, trace=trace)

    if not args.no_log:
        path = os.path.expanduser(args.log or log_path_for(best["label"]))
        written = write_verification_log(path, text, best, trace, raw_bits,
                                         args.decimals)
        if written:
            print(tag("+", f"Journal de vérification : {os.path.abspath(written)}"))

    solo = [r for r in results if r["category"] == "solo"]
    if solo:
        print()
        print(tag("*", "Efficacité des techniques en solo (sans entropy coding) :"))
        for r in sorted(solo, key=size_of):
            size = size_of(r)
            print(f"    • {r['label']:<25} → {trunc_dec(size, 8, 1)} octets  "
                  f"({gain_str(raw_bits, size, 1)} %)")

if __name__ == "__main__":
    main()