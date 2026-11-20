n = int(input("entrer un nombre n:"))
som = 0
for i in range(1,n+1):
    som += i*i*i
print("la somme cubique des nombres entiers est : ", som)
