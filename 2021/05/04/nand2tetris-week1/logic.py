
def f1(x, y, z):
  return (not x) and (not y) and (not z)


def f2(x, y, z):
  return (not x) and y and (not z)


def f3(x, y, z):
  return (not x) and (not y) and (z)


l = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]

print("f1 : ")
for ll in l:
  print(int(f1(*ll)))

print("f2 : ")
for ll in l:
  print(int(f2(*ll)))

print("f3 : ")
for ll in l:
  print(int(f3(*ll)))

# print("f1 or f2 : ")
# for ll in l:
#   print(int(f1(*ll) or f2(*ll)))
