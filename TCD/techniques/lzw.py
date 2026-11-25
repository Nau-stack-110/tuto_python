#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LZW : dictionnaire, codage, décodage et vérification sans perte."""

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


def build_alphabet(text):
    """Symboles distincts, dans leur ordre d'apparition."""
    alphabet = []
    for sym in text:
        if sym not in alphabet:
            alphabet.append(sym)
    return alphabet


def lzw_encode(text):
    """Retourne les codes émis, les étapes du dictionnaire et le final."""
    alphabet = build_alphabet(text)
    dictionary = {sym: index for index, sym in enumerate(alphabet, start=1)}
    next_index = len(alphabet) + 1
    codes, steps = [], []
    for index, sym in enumerate(alphabet, start=1):
        steps.append({"index": index, "entry": sym, "code": None})
    current = ""
    for sym in text:
        candidate = current + sym
        if candidate in dictionary:
            current = candidate
            continue
        code = dictionary[current]
        codes.append(code)
        dictionary[candidate] = next_index
        steps.append({
            "index": next_index,
            "entry": candidate,
            "code": code,
        })
        next_index += 1
        current = sym
    if current:
        code = dictionary[current]
        codes.append(code)
        steps.append({"index": None, "entry": None, "code": code})
    return codes, steps, dictionary, alphabet


def lzw_decode(codes, alphabet):
    """Décode les codes et reconstruit le dictionnaire pas à pas."""
    dictionary = {index: sym for index, sym in enumerate(alphabet, start=1)}
    next_index = len(alphabet) + 1
    out, steps = [], []
    for index, sym in enumerate(alphabet, start=1):
        steps.append({"index": index, "entry": sym, "decoded": None})
    previous = None
    for code in codes:
        if code in dictionary:
            entry = dictionary[code]
        elif previous is not None and code == next_index:
            # Cas KwKwK : le code désigne l'entrée en cours de création.
            entry = previous + previous[0]
        else:
            raise ValueError(f"Code LZW invalide : {code}")
        out.append(entry)
        if previous is None:
            # Premier code : il correspond à une entrée initiale.
            for step in steps:
                if step["index"] == code:
                    step["decoded"] = entry
                    break
        else:
            new_entry = previous + entry[0]
            dictionary[next_index] = new_entry
            steps.append({
                "index": next_index,
                "entry": new_entry,
                "decoded": entry,
            })
            next_index += 1
        previous = entry
    return "".join(out), steps


def main():
    parser = base_parser(
        "LZW · dictionnaire, codage, décodage et vérification",
        """
Exemples :
  python3 techniques/lzw.py
  python3 techniques/lzw.py "ABABABA"
  python3 techniques/lzw.py --no-color -d 4
""",
    )
    args = parser.parse_args()
    prepare(args)
    text = load_text(args, DEFAULT_TEXT)
    raw_bits = show_header("LZW · CODAGE ET DÉCODAGE", text)

    codes, encode_steps, dictionary, alphabet = lzw_encode(text)
    initial = ", ".join(f"{index}→{short_repr(sym, 8)}"
                        for index, sym in enumerate(alphabet, start=1))
    print(tag("*", f"Alphabet initial : {initial}"))

    rows = []
    for step in encode_steps:
        index = "—" if step["index"] is None else str(step["index"])
        entry = "—" if step["entry"] is None else short_repr(step["entry"], 36)
        code = "—" if step["code"] is None else str(step["code"])
        rows.append([index, entry, code])
    render_table(
        "TABLE LZW · ENCODAGE",
        ["Indexe", "Dictionnaire", "Encodage"],
        rows,
        ["right", "left", "right"],
    )

    decoded, decode_steps = lzw_decode(codes, alphabet)
    rows = []
    for step in decode_steps:
        index = "—" if step["index"] is None else str(step["index"])
        entry = "—" if step["entry"] is None else short_repr(step["entry"], 36)
        output = "—" if step["decoded"] is None else short_repr(step["decoded"], 36)
        rows.append([index, entry, output])
    render_table(
        "TABLE LZW · DÉCODAGE",
        ["Indexe", "Dictionnaire", "Décodage"],
        rows,
        ["right", "left", "left"],
    )

    width = bits_for(max(codes))
    code_bits = len(codes) * width
    alphabet_bits = sum(sym_bits(sym) for sym in alphabet)
    payload = code_bits + alphabet_bits
    render_table(
        "RÉSUMÉ DES BITS",
        ["Grandeur", "Valeur", "Calcul"],
        [
            ["Codes", str(len(codes)), "un code par entrée émise"],
            ["Bits par code", f"{width} bits", "bits(index max)"],
            ["Bits des codes", f"{code_bits} bits", f"{len(codes)} × {width}"],
            ["Alphabet initial", f"{alphabet_bits} bits", "Σ bits(symbole)"],
            ["Payload LZW", f"{payload} bits",
             f"{code_bits} + {alphabet_bits}"],
        ],
        ["left", "right", "left"],
    )

    ok = decoded == text
    compressed = " ".join(str(code) for code in codes)
    print(tag("*", f"Codes : {short_repr(codes, 180)}"))
    show_result(raw_bits, payload, ok, args.decimals, text, compressed)


if __name__ == "__main__":
    main()
