temp = []
try:
    n = int(input("entrer le nombre de température : "))
    for i in range(n):
        x = int(input(f"entrer les températures {i} :"))
        temp.append(x)
    print(temp if len(temp) > 0 else 0)

    akaiky_zero = temp[0]
    for t in temp:
        if abs(t) < abs(akaiky_zero):
            akaiky_zero = t
        elif abs(t) == abs(akaiky_zero) and t > akaiky_zero:
            akaiky_zero = t
    print(akaiky_zero)
except:
    print('Misy erreur')
