import math
from mpi4py import MPI
from viztracer import log_sparse
import numpy as np
np.set_printoptions(suppress=True)


@log_sparse(stack_depth=3)
def summa_2_5d(A: np.ndarray, B: np.ndarray, C: np.ndarray, p, d, comm3d: MPI.Cartcomm):
  x_comm = comm3d.Sub([False, False, True])
  y_comm = comm3d.Sub([False, True, False])
  z_comm = comm3d.Sub([True, False, False])

  (M, K) = A.shape
  N = B.shape[1]
  (MTile, NTile, KPTile, KDTile) = (M // p, N // p, K // p, K // d)
  (l, j, i) = comm3d.Get_coords(rank)  # topology is row major
  # align data
  a = np.ascontiguousarray(A[i * MTile:(i + 1) * MTile, j * KPTile:(j + 1)
                           * KPTile]) if l == 0 else np.ones([MTile, KPTile], np.float32)
  b = np.ascontiguousarray(B[i * KPTile:(i + 1) * KPTile, j * NTile:(j + 1)
                           * NTile]) if l == 0 else np.ones([KPTile, NTile], np.float32)
  c = np.zeros([MTile, NTile], dtype=float)
  # compute and passing data

  z_comm.Bcast(a, root=0)
  z_comm.Bcast(b, root=0)
  ktile = math.gcd(KPTile, KDTile)
  for k in range((l * KDTile) // ktile, ((l + 1) * KDTile) // ktile):
    aroot = ((k * ktile) // KPTile)
    Atemp = np.copy(a[:, (k * ktile) - (aroot * KPTile):
                      ((k + 1) * ktile) - (aroot * KPTile)]) if aroot == j else np.empty([MTile, ktile], np.float32)
    y_comm.Bcast(Atemp, root=aroot)

    broot = ((k * ktile) // KPTile)
    Btemp = np.copy(b[(k * ktile) - (broot * KPTile):
                      ((k + 1) * ktile) - (broot * KPTile), :]) if broot == i else np.empty([ktile, NTile], np.float32)
    x_comm.Bcast(Btemp, root=broot)
    np.add(c, np.dot(Atemp, Btemp), out=c)

  cr = np.empty([MTile, NTile], dtype=float) if l == 0 else None
  z_comm.Reduce(c, cr)

  # compare result
  if l == 0:
    ref = C[i * MTile:(i + 1) * MTile, j * NTile:(j + 1) * NTile]
    assert np.allclose(cr, ref)


if __name__ == '__main__':
  A, B, C = np.load('A.npy', 'r'), np.load('B.npy', 'r'), np.load('C.npy', 'r')
  p = 3
  d = 2
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  comm3d = comm.Create_cart([d, p, p], [False, False, False])
  summa_2_5d(A, B, C, p, d, comm3d)
