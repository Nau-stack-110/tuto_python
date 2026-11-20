#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BWT : rotations, tri, dernière colonne, index primaire et inversion."""

try:
    from ._common import (
        base_parser, load_text, prepare, render_table, short_repr,
        show_header, show_result, tag,
    )
except ImportError:  # exécution directe
    from _common import (
        base_parser, load_text, prepare, render_table, short_repr,
        show_header, show_result, tag,
    )

DEFAULT_TEXT = "banana"


def bwt_transform(text):
    n = len(text)
    rotations = [(i, text[i:] + text[:i]) for i in range(n)]
    ordered = sorted(rotations, key=lambda item: item[1])
    last = "".join(rotation[-1] for _, rotation in ordered)
    primary = next(rank for rank, (origin, _) in enumerate(ordered) if origin == 0)
    return rotations, ordered, last, primary


def inverse_bwt(last, primary):
    n = len(last)
    if n == 0:
        return ""
    # next[i] = ligne de F qui correspond à L[i]
    nxt = sorted(range(n), key=lambda i: last[i])
    first = [last[i] for i in nxt]
    out, row = [], primary
    for _ in range(n):
        out.append(first[row])
        row = nxt[row]
    return "".join(out)


def main():
    parser = base_parser(
        "BWT · rotations possibles, tri, dernière colonne et reconstruction",
        """
Exemples :
  python3 techniques/bwt.py
  python3 techniques/bwt.py "banana"
  python3 techniques/bwt.py --max-rotations 20 --no-color
""",
    )
    parser.add_argument("--max-rotations", type=int, default=12,
                        help="Nombre max de rotations affichées (défaut : 12)")
    args = parser.parse_args()
    prepare(args)
    text = load_text(args, DEFAULT_TEXT)
    raw_bits = show_header("BWT · BURROWS-WHEELER TRANSFORM", text)

    rotations, ordered, last, primary = bwt_transform(text)
    n = len(text)

    shown = rotations[:args.max_rotations]
    render_table(
        f"ROTATIONS POSSIBLES ({len(shown)}/{len(rotations)})"
        if len(shown) < len(rotations) else "ROTATIONS POSSIBLES",
        ["#", "Rotation"],
        [[str(i), short_repr(rotation, 64)] for i, rotation in shown],
        ["right", "left"],
    )

    shown_ordered = ordered[:args.max_rotations]
    render_table(
        f"ROTATIONS TRIÉES ({len(shown_ordered)}/{len(ordered)})"
        if len(shown_ordered) < len(ordered) else "ROTATIONS TRIÉES",
        ["Rang", "Origine", "Rotation", "Dernière colonne"],
        [[str(rank), str(origin), short_repr(rotation, 56), short_repr(rotation[-1], 12)]
         for rank, (origin, rotation) in enumerate(shown_ordered)],
        ["right", "right", "left", "center"],
    )

    decoded = inverse_bwt(last, primary)
    ok = decoded == text
    payload = n * 8

    print(tag("*", f"Dernière colonne : {short_repr(last, 120)}"))
    print(tag("*", f"Index primaire : {primary}"))
    print(tag("*", f"Payload BWT = {n} × 8 = {payload} bits (transformation réversible, "
                   f"pas de réduction à elle seule)"))
    show_result(raw_bits, payload, ok, args.decimals, text, last)


if __name__ == "__main__":
    main()
