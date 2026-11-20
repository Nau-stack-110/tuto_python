#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Outils partagés par les démonstrations des techniques de compression."""

import argparse
import os
import sys


class Ansi:
    enabled = False


def paint(text, *codes):
    if not Ansi.enabled:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"


def tag(kind, msg):
    colors = {"*": "36", "+": "32", "!": "33", "x": "31"}
    return f"{paint(f'[{kind}]', '1', colors.get(kind, '37'))} {msg}"


def trunc_dec(num, den, d=2):
    """Division entière affichée avec une virgule, sans flottants."""
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


def gain_str(raw, total, d=2):
    if raw >= total:
        return trunc_dec((raw - total) * 100, raw, d)
    return "-" + trunc_dec((total - raw) * 100, raw, d)


def bits_for(x):
    return max(1, int(x).bit_length()) if x >= 0 else 1


def sym_bits(sym):
    if isinstance(sym, str):
        return 8 * max(1, len(sym.encode("utf-8")))
    if isinstance(sym, (list, tuple)):
        return 8 + sum(sym_bits(x) for x in sym)
    if isinstance(sym, int):
        return max(1, (abs(sym).bit_length() + 7) // 8) * 8
    return 32


def render_table(title, headers, rows, aligns=None):
    """Tableau encadré, dans le style des supports de cours."""
    aligns = aligns or ["left"] * len(headers)
    real = [r for r in rows if r is not None]
    widths = [max([len(headers[i])] + [len(str(r[i])) for r in real])
              for i in range(len(headers))]

    def line(left, mid, right):
        return paint(left + mid.join("─" * (w + 2) for w in widths) + right,
                     "2", "32")

    def pad(value, index):
        text = str(value)
        if aligns[index] == "right":
            return text.rjust(widths[index])
        if aligns[index] == "center":
            return text.center(widths[index])
        return text.ljust(widths[index])

    bar = paint("│", "2", "32")
    print(f"\n{tag('+', title)}")
    print(line("┌", "┬", "┐"))
    print(bar + bar.join(" " + paint(h.center(widths[i]), "1", "32") + " "
                         for i, h in enumerate(headers)) + bar)
    print(line("├", "┼", "┤"))
    for row in rows:
        if row is None:
            print(bar + bar.join(" " + paint("⋮".center(w), "2") + " "
                                 for w in widths) + bar)
        else:
            print(bar + bar.join(f" {pad(v, i)} " for i, v in enumerate(row)) + bar)
    print(line("└", "┴", "┘"))


def short_repr(value, limit=160):
    text = repr(value)
    return text if len(text) <= limit else text[:limit - 1] + "…"


def short_bits(bits, limit=160):
    return bits if len(bits) <= limit else bits[:limit - 1] + f"… ({len(bits)} bits)"


def short_text(value, limit=90):
    text = str(value)
    return text if len(text) <= limit else text[:limit - 1] + "…"


def short_value(value, limit=48):
    text = str(value)
    return text if len(text) <= limit else text[:limit - 1] + "…"


def base_parser(description, epilog=""):
    ap = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    ap.add_argument("text", nargs="?", default=None,
                    help="Texte ou chemin de fichier")
    ap.add_argument("-f", "--file", metavar="PATH", help="Lire depuis un fichier")
    ap.add_argument("-d", "--decimals", type=int, default=2,
                    help="Chiffres après la virgule (défaut : 2)")
    ap.add_argument("--no-color", action="store_true",
                    help="Désactiver les couleurs")
    return ap


def prepare(args):
    Ansi.enabled = sys.stdout.isatty() and not args.no_color


def load_text(args, default):
    if args.file:
        if not os.path.isfile(args.file):
            print(tag("x", f"Fichier introuvable : {args.file}"), file=sys.stderr)
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    elif args.text is not None:
        if os.path.isfile(args.text):
            with open(args.text, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        else:
            text = args.text
    elif not sys.stdin.isatty():
        data = sys.stdin.read()
        text = data if data.strip() else default
    else:
        text = default
    text = text.rstrip("\n")
    if not text:
        print(tag("x", "Texte vide – rien à compresser."), file=sys.stderr)
        sys.exit(1)
    return text


def show_header(title, text):
    raw_bits = len(text.encode("utf-8")) * 8
    print()
    print(paint("=" * 72, "2", "36"))
    print(paint(f" {title}", "1", "36"))
    print(paint("=" * 72, "2", "36"))
    print(tag("*", f"Texte original : {short_repr(text)}"))
    print(tag("*", f"{len(text)} caractères → "
                   f"{len(text.encode('utf-8'))} octets ({raw_bits} bits)"))
    return raw_bits


def result_rows(raw_bits, payload_bits, ok, decimals,
                original, compressed_bits=None):
    size = payload_bits
    rows = [
        ["Texte original", short_repr(original)],
        ["Bits originaux", f"{raw_bits} bits"],
        ["Texte compressé", short_text(compressed_bits or "", 90)],
        ["Payload compressé", f"{size} bits ≈ {trunc_dec(size, 8, 2)} octets"],
        ["Taux de réduction", f"{gain_str(raw_bits, size, decimals)} %"],
        ["Vérification", "OK" if ok else "ÉCHEC"],
    ]
    return rows


def show_result(raw_bits, payload_bits, ok, decimals,
                original, compressed_bits=None):
    rows = result_rows(raw_bits, payload_bits, ok,
                       decimals, original, compressed_bits)
    render_table(
        "RÉSULTAT (payload seul)",
        ["Élément", "Valeur"],
        rows,
        ["left", "left"],
    )
    if ok:
        print(tag("+", "Vérification sans perte : OK"))
    else:
        print(tag("x", "Vérification sans perte : ÉCHEC"))
