#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LZ77 : buffer de recherche, buffer d'entrée, tokens et vérification."""

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

DEFAULT_TEXT = "abracadabra"


def find_match(text, pos, window, lookahead):
    """Plus longue correspondance non chevauchante dans la fenêtre."""
    best_length, best_distance = 0, 0
    for start in range(max(0, pos - window), pos):
        max_length = min(lookahead, len(text) - pos, pos - start)
        length = 0
        while (length < max_length
               and text[start + length] == text[pos + length]):
            length += 1
        if length > best_length:
            best_length, best_distance = length, pos - start
    return best_distance, best_length


def token_bits(token):
    if token[0] == "L":
        return 1 + sym_bits(token[1])
    bits = 1 + bits_for(token[1]) + bits_for(token[2])
    if token[3]:
        bits += sym_bits(token[3])
    return bits


def lz77_encode(text, window, lookahead):
    tokens, steps, pos = [], [], 0
    while pos < len(text):
        search = text[max(0, pos - window):pos]
        look = text[pos:pos + lookahead]
        distance, length = find_match(text, pos, window, lookahead)
        if length >= 3:
            next_symbol = text[pos + length] if pos + length < len(text) else ""
            token = ("M", distance, length, next_symbol)
            consumed = length + (1 if next_symbol else 0)
        else:
            token = ("L", text[pos])
            consumed = 1
        steps.append((pos, search, look, token))
        tokens.append(token)
        pos += consumed
    return tokens, steps


def lz77_decode(tokens):
    out = []
    for token in tokens:
        if token[0] == "L":
            out.append(token[1])
        else:
            distance, length, next_symbol = token[1], token[2], token[3]
            start = len(out) - distance
            for offset in range(length):
                out.append(out[start + offset])
            if next_symbol:
                out.append(next_symbol)
    return "".join(out)


def main():
    parser = base_parser(
        "LZ77 · buffer de recherche, buffer d'entrée et tokens",
        """
Exemples :
  python3 techniques/lz77.py
  python3 techniques/lz77.py "abracadabra"
  python3 techniques/lz77.py --window 8 --lookahead 5 --no-color
""",
    )
    parser.add_argument("--window", type=int, default=8,
                        help="Taille du buffer de recherche (défaut : 8)")
    parser.add_argument("--lookahead", type=int, default=5,
                        help="Taille du buffer d'entrée (défaut : 5)")
    args = parser.parse_args()
    prepare(args)
    text = load_text(args, DEFAULT_TEXT)
    raw_bits = show_header("LZ77 · FENÊTRE GLISSANTE", text)

    tokens, steps = lz77_encode(text, max(1, args.window), max(1, args.lookahead))
    rows = []
    for index, (pos, search, look, token) in enumerate(steps, 1):
        if token[0] == "L":
            distance = length = "—"
            next_symbol = "—"
        else:
            distance, length = str(token[1]), str(token[2])
            next_symbol = short_repr(token[3], 12) if token[3] else "fin"
        rows.append([
            str(index),
            str(pos),
            short_repr(search, 30) if search else "∅",
            short_repr(look, 30) if look else "∅",
            short_repr(token, 30),
            distance,
            length,
            next_symbol,
            str(token_bits(token)),
        ])
    render_table(
        f"TABLE LZ77 (fenêtre={args.window}, entrée={args.lookahead})",
        ["#", "Position", "Buffer de recherche", "Buffer d'entrée",
         "Token", "Distance", "Longueur", "Suivant", "Bits"],
        rows,
        ["right", "right", "left", "left", "left",
         "right", "right", "center", "right"],
    )

    payload = sum(token_bits(token) for token in tokens)
    literals = sum(1 for token in tokens if token[0] == "L")
    matches = len(tokens) - literals
    literal_bits = sum(token_bits(token) for token in tokens if token[0] == "L")
    match_bits = payload - literal_bits
    render_table(
        "RÉSUMÉ DES TOKENS",
        ["Grandeur", "Valeur", "Calcul"],
        [
            ["Littéraux", str(literals), f"{literal_bits} bits"],
            ["Copies", str(matches), f"{match_bits} bits"],
            ["Payload LZ77", f"{payload} bits", f"{literal_bits} + {match_bits}"],
        ],
        ["left", "right", "left"],
    )

    decoded = lz77_decode(tokens)
    ok = decoded == text
    compressed = " ".join(short_repr(token, 24) for token in tokens)
    print(tag("*", f"Tokens : {short_repr(tokens, 160)}"))
    show_result(raw_bits, payload, ok, args.decimals, text, compressed)


if __name__ == "__main__":
    main()
