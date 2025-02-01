# s = str(input())
# vo = "AOIEUaoeiu"
# c = {v: 0 for v in "AOEIU"}
# for i in s:
#     if i in vo:
#         c[i.upper()] += 1
# print(c)


# class Solution
# {
#     static void Main(string[] args)
#     {
#         string s = Console.ReadLine().ToUpper();

#         Console.WriteLine($"{s.Count(a=>a=='A')} {s.Count(a=>a=='E')} {s.Count(a=>a=='I')} {s.Count(a=>a=='O')} {s.Count(a=>a=='U')}");
#     }
# }
        
s = input()
vo = "AOEIU"
c = {v:0 for v in vo}
for i in s.upper():
    if i in c:
        c[i] += 1
print("".join(str(c[v]) for v in vo))
              