import sys
import math
d={
1:'Mistake',
2:'Very bad',
3:'Bad',
4:'So so',
5:'Average',
6:'Quite good',
7:'Good',
8:'Very good',
9:'Sensational',
10:'Masterpiece!',
}
n = int(input())
a=0
for i in range(n):
    r = input()
    a+=r.count('*')
b=round(a/n)

print(d[b])

