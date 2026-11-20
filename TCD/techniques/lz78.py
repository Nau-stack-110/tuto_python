#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LZ78 : dictionnaire, tokens (index, symbole) et vérification."""

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

DEFAULT_TEXT = "ABABABA"


def lz78_encode(text):
    dictionary = {(): 0}
    next_index = 1
    current = ()
    tokens, steps = [], []
    for sym in text:
        candidate = current + (sym,)
        if candidate in dictionary:
            current = candidate
        else:
            index = dictionary[current]
            tokens.append(("D", index, sym))
            steps.append({
                "phrase": "".join(current),
                "symbol": sym,
                "index": index,
                "entry": "".join(candidate),
                "new_index": next_index,
            })
            dictionary[candidate] = next_index
            next_index += 1
            current = ()
    if current:
        index = dictionary[current]
        tokens.append(("D", index, None))
        steps.append({
            "phrase": "".join(current),
            "symbol": None,
            "index": index,
            "entry": "".join(current),
            "new_index": None,
        })
    return tokens, steps, dictionary


def lz78_decode(tokens):
    dictionary = {0: ""}
    next_index = 1
    out = []
    for _, index, sym in tokens:
        entry = dictionary[index] + (sym or "")
        out.append(entry)
        dictionary[next_index] = entry
        next_index += 1
    return "".join(out)


def token_bits(token):
    index, sym = token[1], token[2]
    return bits_for(index) + (1 if sym is None else sym_bits(sym))


def main():
    parser = base_parser(
        "LZ78 · dictionnaire, tokens et vérification",
        """
Exemples :
  python3 techniques/lz78.py
  python3 techniques/lz78.py "ABABABA"
  python3 techniques/lz78.py --no-color -d 4
""",
    )
    args = parser.parse_args()
    prepare(args)
    text = load_text(args, DEFAULT_TEXT)
    raw_bits = show_header("LZ78 · DICTIONNAIRE ET TOKENS", text)

    tokens, steps, dictionary = lz78_encode(text)

    rows = []
    for number, step in enumerate(steps, 1):
        symbol = "fin" if step["symbol"] is None else short_repr(step["symbol"], 12)
        new_index = "—" if step["new_index"] is None else str(step["new_index"])
        rows.append([
            str(number),
            short_repr(step["phrase"], 28) if step["phrase"] else "ε",
            symbol,
            short_repr(("D", step["index"], step["symbol"]), 28),
            short_repr(step["entry"], 32),
            new_index,
            str(token_bits(tokens[number - 1])),
        ])
    render_table(
        "TABLE LZ78 · CONSTRUCTION DU DICTIONNAIRE",
        ["#", "Phrase w", "Symbole c", "Token", "Nouvelle entrée", "Nouvel index", "Bits"],
        rows,
        ["right", "left", "center", "left", "left", "right", "right"],
    )

    dictionary_rows = []
    for entry_tuple in sorted(dictionary, key=lambda item: dictionary[item]):
        index = dictionary[entry_tuple]
        entry = "".join(entry_tuple)
        dictionary_rows.append([
            str(index),
            short_repr(entry, 40) if entry else "ε",
            str(len(entry_tuple)),
        ])
    render_table(
        "DICTIONNAIRE FINAL",
        ["Index", "Entrée", "Longueur"],
        dictionary_rows,
        ["right", "left", "right"],
    )

    max_index = max((token[1] for token in tokens), default=0)
    index_bits = sum(bits_for(token[1]) for token in tokens)
    symbol_bits = sum(1 if token[2] is None else sym_bits(token[2])
                      for token in tokens)
    table_bits = 2 * bits_for(max_index)
    payload = index_bits + symbol_bits + table_bits

    render_table(
        "RÉSUMÉ DES BITS",
        ["Grandeur", "Valeur", "Calcul"],
        [
            ["Tokens", str(len(tokens)), "un token = (index, symbole)"],
            ["Bits index", f"{index_bits} bits", "Σ bits(index)"],
            ["Bits symboles", f"{symbol_bits} bits", "Σ bits(symbole)"],
            ["Table", f"{table_bits} bits", "2 × bits(index max)"],
            ["Payload LZ78", f"{payload} bits",
             f"{index_bits} + {symbol_bits} + {table_bits}"],
        ],
        ["left", "right", "left"],
    )

    decoded = lz78_decode(tokens)
    ok = decoded == text
    compressed = " ".join(short_repr(token, 22) for token in tokens)
    print(tag("*", f"Tokens : {short_repr(tokens, 180)}"))
    show_result(raw_bits, payload, ok, args.decimals, text, compressed)


if __name__ == "__main__":
    main()
