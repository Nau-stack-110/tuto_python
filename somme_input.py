n= str(input())
s = 0
# print("longueur : ", len(n))
for i in range(len(n)):
    s += int(n[i])
print(s)