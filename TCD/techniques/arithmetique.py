#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Codage arithmétique : fréquences, intervalles L_j/R_j et code binaire."""

from collections import Counter
from fractions import Fraction

try:
    from ._common import (
        base_parser, load_text, prepare, render_table, short_bits,
        short_repr, show_header, show_result, tag,
    )
except ImportError:  # exécution directe
    from _common import (
        base_parser, load_text, prepare, render_table, short_bits,
        short_repr, show_header, show_result, tag,
    )

DEFAULT_TEXT = "BANANA"


def fraction_str(value):
    return f"{value.numerator}/{value.denominator}"


def trunc_dec4(value, digits=4):
    """Tronque une fraction positive à `digits` chiffres après la virgule."""
    scale = 10 ** digits
    q = (value.numerator * scale) // value.denominator
    return f"{q // scale},{q % scale:0{digits}d}"


def sci_str(value, digits=4):
    """Notation scientifique décimale, sans passer par un flottant."""
    if value == 0:
        return "0"
    num, den = value.numerator, value.denominator
    exponent = int((num.bit_length() - den.bit_length()) * 0.3010299956639812)
    if exponent >= 0:
        while num < den * 10 ** exponent:
            exponent -= 1
        while num >= den * 10 ** (exponent + 1):
            exponent += 1
        scaled_num, scaled_den = num, den * 10 ** exponent
    else:
        while num * 10 ** (-exponent) < den:
            exponent -= 1
        while num * 10 ** (-exponent - 1) >= den:
            exponent += 1
        scaled_num, scaled_den = num * 10 ** (-exponent), den
    scale = 10 ** digits
    q = (2 * scaled_num * scale + scaled_den) // (2 * scaled_den)
    return f"{q // scale},{q % scale:0{digits}d}e{exponent:+d}"


def short_fraction(value, limit=48):
    if (value.numerator.bit_length() <= 120
            and value.denominator.bit_length() <= 120):
        text = fraction_str(value)
        if len(text) <= limit:
            return text
    return sci_str(value, 4)


def short_int(value, limit=32):
    try:
        text = str(value)
    except ValueError:
        return f"<{value.bit_length()} bits>"
    if len(text) <= limit:
        return text
    half = max(1, (limit - 1) // 2)
    return text[:half] + "…" + text[-half:]


def build_frequencies(text):
    counts = Counter(text)
    alphabet = list(dict.fromkeys(text))
    total = len(text)
    probabilities = {sym: Fraction(counts[sym], total) for sym in alphabet}
    cumulative = {}
    current = Fraction(0)
    rows = []
    for index, sym in enumerate(alphabet, 1):
        cumulative[sym] = current
        upper = current + probabilities[sym]
        rows.append([
            str(index),
            short_repr(sym, 16),
            str(counts[sym]),
            trunc_dec4(probabilities[sym], 4),
            trunc_dec4(current, 4),
            trunc_dec4(upper, 4),
        ])
        current = upper
    return alphabet, probabilities, cumulative, rows


def arithmetic_steps(text, alphabet, probabilities, cumulative):
    index = {sym: i for i, sym in enumerate(alphabet, 1)}
    low, width = Fraction(0), Fraction(1)
    rows = [["0", "—", "départ", trunc_dec4(low, 4), trunc_dec4(width, 4)]]
    for j, sym in enumerate(text, 1):
        low_new = low + width * cumulative[sym]
        width_new = width * probabilities[sym]
        rows.append([
            str(j),
            str(index[sym]),
            short_repr(sym, 16),
            trunc_dec4(low_new, 4),
            trunc_dec4(low_new + width_new, 4),
        ])
        low, width = low_new, width_new
    return low, width, rows


def choose_midpoint_code(low, width):
    """Code binaire le plus court, choisi autour de (Ln + Rn) / 2."""
    if width <= 0:
        raise ValueError("Intervalle vide : aucun code possible.")
    high = low + width
    mid = low + width / 2

    # Point de départ : plus petit k tel que 2^k * largeur > 1.
    k = max(1, (width.denominator // width.numerator).bit_length())
    # Descendre tant qu'un code plus court tombe encore dans l'intervalle.
    while k > 1:
        scale = 1 << (k - 1)
        first = (low.numerator * scale) // low.denominator + 1
        if Fraction(first, scale) < high:
            k -= 1
        else:
            break

    scale = 1 << k
    first = (low.numerator * scale) // low.denominator + 1
    left = (mid.numerator * scale) // mid.denominator
    candidates = {first - 1, first, left, left + 1}

    best = None
    for m in candidates:
        value = Fraction(m, scale)
        if low < value < high:
            if best is None or abs(value - mid) < abs(best[1] - mid):
                best = (m, value)
    if best is None:
        raise ValueError("Impossible de trouver un code binaire dans l'intervalle.")
    return best[0], k, mid, best[1]


def arithmetic_decode(code, k, alphabet, probabilities, cumulative, length):
    value = Fraction(code, 1 << k)
    out = []
    for _ in range(length):
        for sym in alphabet:
            low = cumulative[sym]
            high = low + probabilities[sym]
            if low <= value < high:
                out.append(sym)
                value = (value - low) / (high - low)
                break
        else:
            raise ValueError("Code arithmétique hors intervalle.")
    return "".join(out)


def main():
    parser = base_parser(
        "Arithmétique · table des fréquences, réduction de l'intervalle et code",
        """
Exemples :
  python3 techniques/arithmetique.py
  python3 techniques/arithmetique.py "BANANA"
  python3 techniques/arithmetique.py --no-color -d 4
""",
    )
    args = parser.parse_args()
    prepare(args)
    text = load_text(args, DEFAULT_TEXT)
    raw_bits = show_header("CODAGE ARITHMÉTIQUE · INTERVALLES", text)

    alphabet, probabilities, cumulative, freq_rows = build_frequencies(text)
    render_table(
        "TABLE 1 · FRÉQUENCES ET PROBABILITÉS CUMULÉES",
        ["i", "s_i", "n_i", "P(s_i)", "C(s_i-1)", "C(s_i)"],
        freq_rows,
        ["right", "left", "right", "right", "right", "right"],
    )

    low, width, step_rows = arithmetic_steps(
        text, alphabet, probabilities, cumulative
    )
    render_table(
        "TABLE 2 · RÉDUCTION DE L'INTERVALLE (L_j ET R_j)",
        ["j", "i", "donnée", "L_j", "R_j"],
        step_rows,
        ["right", "right", "left", "right", "right"],
    )

    m, k, mid, code_value = choose_midpoint_code(low, width)
    bits = format(m, f"0{k}b")
    high = low + width
    render_table(
        "CODE FINAL",
        ["m", "k", "Code binaire", "Milieu M", "v = m/2^k"],
        [[
            short_int(m, 40),
            f"{k} bits",
            short_bits(bits, 72),
            short_fraction(mid),
            short_fraction(code_value),
        ]],
        ["right", "right", "center", "right", "right"],
    )
    print(tag("*", f"k = {k} bits : plus petit k contenant un code dans ]L_n ; R_n["))
    print(tag("*", f"L_n = {short_fraction(low)} ; "
                   f"R_n = {short_fraction(high)} ; "
                   f"largeur = {short_fraction(width)}"))
    print(tag("*", f"Milieu M = (L_n + R_n) / 2 = {short_fraction(mid)} ; "
                   f"code retenu v = {short_fraction(code_value)}"))

    decoded = arithmetic_decode(m, k, alphabet, probabilities, cumulative, len(text))
    ok = decoded == text
    show_result(raw_bits, k, ok, args.decimals, text, bits)


if __name__ == "__main__":
    main()
