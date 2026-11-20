a = int(input("entrer l'année :"))
if (a % 4 == 0 and a % 100 != 0) or a % 400 == 0:
    print(f"{a} est année bissextile")
else:
    print(f"{a} n'est pas un année bissextile")
