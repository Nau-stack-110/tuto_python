#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Huffman : fréquences, codes, texte binaire, taille et taux de réduction."""

import heapq
from collections import Counter

try:
    from ._common import (
        base_parser, load_text, prepare, render_table,
        short_repr, show_header, show_result, tag, trunc_dec,
    )
except ImportError:  # exécution directe : python3 techniques/huffman.py
    from _common import (
        base_parser, load_text, prepare, render_table,
        short_repr, show_header, show_result, tag, trunc_dec,
    )

DEFAULT_TEXT = "ABRACADABRA"


def build_tree(freqs, order):
    heap = [(count, order[sym], ("L", sym)) for sym, count in freqs.items()]
    heapq.heapify(heap)
    merges = []
    serial = len(order)
    while len(heap) > 1:
        count_a, _, node_a = heapq.heappop(heap)
        count_b, _, node_b = heapq.heappop(heap)
        merges.append((node_a, node_b, count_a + count_b))
        heapq.heappush(heap, (count_a + count_b, serial, ("N", node_a, node_b)))
        serial += 1
    return heap[0][2], merges


def read_lengths(node, depth=0, lengths=None):
    if lengths is None:
        lengths = {}
    if node[0] == "L":
        lengths[node[1]] = max(depth, 1)
        return lengths
    read_lengths(node[1], depth + 1, lengths)
    read_lengths(node[2], depth + 1, lengths)
    return lengths


def canonical_codes(lengths, order):
    table = sorted(lengths.items(), key=lambda item: (item[1], order[item[0]]))
    code, previous, codes = 0, table[0][1], {}
    for sym, length in table:
        code <<= length - previous
        previous = length
        codes[sym] = format(code, f"0{length}b")
        code += 1
    return table, codes


def decode(bits, codes):
    inverse = {code: sym for sym, code in codes.items()}
    out, current = [], ""
    for bit in bits:
        current += bit
        if current in inverse:
            out.append(inverse[current])
            current = ""
    return "".join(out), current == ""


def node_label(node):
    if node[0] == "L":
        return short_repr(node[1], 18)
    return f"({node_label(node[1])}+{node_label(node[2])})"


def main():
    parser = base_parser(
        "Huffman · table des symboles, fréquences, codes et résultat binaire",
        """
Exemples :
  python3 techniques/huffman.py
  python3 techniques/huffman.py "ABRACADABRA"
  python3 techniques/huffman.py --no-color -d 4
""",
    )
    args = parser.parse_args()
    prepare(args)
    text = load_text(args, DEFAULT_TEXT)
    raw_bits = show_header("HUFFMAN · CODES ET RÉSULTAT BINAIRE", text)

    counts = Counter(text)
    order = {sym: i for i, sym in enumerate(dict.fromkeys(text))}
    tree, merges = build_tree(counts, order)
    lengths = read_lengths(tree)
    table, codes = canonical_codes(lengths, order)
    bits = "".join(codes[ch] for ch in text)
    payload = len(bits)

    total = len(text)
    rows = []
    for sym, length in table:
        freq = counts[sym]
        rows.append([
            short_repr(sym, 22),
            str(freq),
            trunc_dec(freq, total, 4),
            codes[sym],
            str(length),
            str(freq * length),
        ])
    rows.append(["TOTAL", str(total), "1,0000", "—", "—", str(payload)])
    render_table(
        "TABLE DE HUFFMAN",
        ["Symbole", "Fréquence", "Probabilité", "Code", "Longueur", "Fréq. × longueur"],
        rows,
        ["left", "right", "right", "center", "right", "right"],
    )

    if len(merges) <= 12:
        merge_rows = [
            [str(i), f"{node_label(a)} + {node_label(b)}", str(weight)]
            for i, (a, b, weight) in enumerate(merges, 1)
        ]
        render_table(
            "CONSTRUCTION DE L'ARBRE (FUSIONS)",
            ["Étape", "Fusion", "Poids"],
            merge_rows,
            ["right", "left", "right"],
        )
    else:
        print(tag("*", f"Fusions : {len(merges)} étapes (table non affichée, "
                       f"alphabet trop grand)."))

    decoded, clean = decode(bits, codes)
    ok = decoded == text and clean

    preview = ", ".join(f"{short_repr(s, 12)}={c}"
                       for s, c in list(codes.items())[:10])
    if len(codes) > 10:
        preview += f", … ({len(codes)} symboles)"
    print(tag("*", f"Codes : {preview}"))
    show_result(raw_bits, payload, ok, args.decimals, text, bits)


if __name__ == "__main__":
    main()
