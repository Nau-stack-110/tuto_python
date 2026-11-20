s = input()
def rev_alnum(word):
    letters = [c for c in word if c.isalnum()]
    letters.reverse()
    res = list(word)
    j = 0
    for i, c in enumerate(word):
        if c.isalnum():
            res[i] = letters[j]
            j += 1
    return "".join(res)
print(" ".join(rev_alnum(w) for w in s.split(" ")))