#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RLE : runs, représentation compressée, bits, taille et taux."""

try:
    from ._common import (
        base_parser, bits_for, load_text, prepare, render_table, short_repr,
        show_header, show_result, sym_bits, tag,
    )
except ImportError:  # exécution directe
    from _common import (
        base_parser, bits_for, load_text, prepare, render_table, short_repr,
        show_header, show_result, sym_bits, tag,
    )

DEFAULT_TEXT = "AAAABBBCCDAA"


def rle_encode(text):
    runs = []
    previous, count = text[0], 1
    for sym in text[1:]:
        if sym == previous:
            count += 1
        else:
            runs.append((previous, count))
            previous, count = sym, 1
    runs.append((previous, count))
    return runs


def rle_decode(runs):
    return "".join(sym * count for sym, count in runs)


def main():
    parser = base_parser(
        "RLE · runs, représentation compressée et taux de réduction",
        """
Exemples :
  python3 techniques/rle.py
  python3 techniques/rle.py "AAAABBBCCDAA"
  python3 techniques/rle.py --no-color -d 4
""",
    )
    args = parser.parse_args()
    prepare(args)
    text = load_text(args, DEFAULT_TEXT)
    raw_bits = show_header("RLE · RUN-LENGTH ENCODING", text)

    runs = rle_encode(text)
    rows = []
    payload = 0
    for index, (sym, count) in enumerate(runs, 1):
        symbol_bits = sym_bits(sym)
        length_bits = bits_for(count)
        run_bits = symbol_bits + length_bits
        payload += run_bits
        rows.append([
            str(index),
            short_repr(sym, 18),
            str(count),
            str(symbol_bits),
            str(length_bits),
            str(run_bits),
            f"{short_repr(sym, 12)}×{count}",
        ])
    rows.append(["TOTAL", "—", str(len(text)), "—", "—", str(payload), f"{len(runs)} runs"])
    render_table(
        "TABLE RLE",
        ["#", "Symbole", "Longueur", "Bits symbole", "Bits longueur",
         "Total bits", "Représentation"],
        rows,
        ["right", "left", "right", "right", "right", "right", "left"],
    )

    compressed = " ".join(f"{short_repr(sym, 12)}×{count}" for sym, count in runs)
    decoded = rle_decode(runs)
    ok = decoded == text

    print(tag("*", f"Runs : {len(runs)} ; payload = Σ(bits(symbole) + "
                   f"bits(longueur)) = {payload} bits"))
    show_result(raw_bits, payload, ok, args.decimals, text, compressed)


if __name__ == "__main__":
    main()
