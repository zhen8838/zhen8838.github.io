
import numpy as np
from mpi4py import MPI
import sys
RANK = MPI.COMM_WORLD.Get_rank()
COMM_ALL = MPI.COMM_WORLD.Create_cart([8, 8])


COMM_0 = COMM_ALL.Sub([False, False])
COMM_1 = COMM_ALL.Sub([False, True])
COMM_2 = COMM_ALL.Sub([True, False])
COMM_3 = COMM_ALL.Sub([True, True])
COMM_3.Bcast
(x, y) = pids = COMM_0.Get_coords(RANK) 


class CommBroadcast:
  def __init__(self, axes: list[int]):
    self.axes = axes

  def __call__(self, srcbuf: np.ndarray, source: list[int], recvbuf: np.ndarray, dest: list[int]):
    comm = COMM_ALL.Sub([True if i in self.axes else False for i in range(len(pids))])
    if comm.Get_rank() == source:
      np.copyto(srcbuf, recvbuf)
    comm.Bcast(recvbuf, root=comm.Get_cart_rank(source))


class CommSendrecv:
  def __call__(self, srcbuf: np.ndarray, source: list[int], recvbuf: np.ndarray, dest: list[int]):
    COMM_ALL.Sendrecv(srcbuf, COMM_ALL.Get_cart_rank(dest), 0,
                      recvbuf, COMM_ALL.Get_cart_rank(source))

def 
