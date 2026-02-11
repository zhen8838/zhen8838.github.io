# from __future__ import annotations
from enum import Enum


class color(Enum):
  r = 0x01
  g = 0x02
  b = 0x03


class A():
  c: color.r

print(A.__annotations__)
