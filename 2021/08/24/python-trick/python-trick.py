class Node():
  def __init__(self, op: str, args: tuple) -> None:
    self.op = op
    self.args = args

  def __repr__(self) -> str:
    if self.args:
      return '{' + f" {self.op}({','.join(str(arg) for arg in self.args)}) " + '}'
    else:
      return f' {self.op}() '


node = Node('a', (Node('a' + str(i), ()) for i in range(10)))
node
""" 
{ a( a0() , a1() , a2() , a3() , a4() , a5() , a6() , a7() , a8() , a9() ) }
"""
node
"""{ a() }
"""


class Base():
  def __new__(cls, name, attr):
    obj = super().__new__(cls)
    obj.name = name
    obj.attr = attr
    return obj

  def __init__(self, value = 0) -> None:
    self.value = value

  def __repr__(self) -> str:
    return f'{self.name} {self.attr}: {self.value}'


class Var(Base):
  def __new__(cls, value: int):
    return super().__new__(cls, 'var', 'constant')


Var()
