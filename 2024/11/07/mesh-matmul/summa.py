from mpi4py import MPI
from viztracer import log_sparse
import numpy as np
np.set_printoptions(suppress=True)

def matmul(c: np.ndarray, a, b):
  c += a @ b

@log_sparse(stack_depth=3)
def summa(A: np.ndarray, B: np.ndarray, C: np.ndarray, P, comm2d: MPI.Cartcomm):
  col_comm = comm2d.Sub([True, False]) # [y,x]
  row_comm = comm2d.Sub([False, True])

  (M, K) = A.shape
  N = B.shape[1]
  (mTile, nTile, kTile) = (M // P, N // P, K // P)
  (j, i) = comm2d.Get_coords(rank)  # topology is row major, [y,x]
  # align data
  a = np.ascontiguousarray(A[j * mTile:(j + 1) * mTile, i * kTile:(i + 1) * kTile].copy())
  b = np.ascontiguousarray(B[j * kTile:(j + 1) * kTile, i * nTile:(i + 1) * nTile].copy())

  # compute and broadcast
  c = np.empty([mTile, nTile], np.float32)
  for k in range(P):
    Atemp = a if k == i else np.empty_like(a)
    Btemp = b if k == j else np.empty_like(b)
    row_comm.Bcast(Atemp, root=k) # A是 [m@x,k@y]，把y的数据进行分发时，需要把[y@false,x@true]，也就是另一路设置为true。
    col_comm.Bcast(Btemp, root=k)
    matmul(c.view(), Atemp, Btemp)

  # compare result
  ref = C[j * mTile:(j + 1) * mTile, i * nTile:(i + 1) * nTile]
  assert np.allclose(c, ref)


if __name__ == '__main__':
  A, B, C = np.load('A.npy', 'r'), np.load('B.npy', 'r'), np.load('C.npy', 'r')
  P = 3
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  comm2d = comm.Create_cart([P, P], [False, False])
  summa(A, B, C, P, comm2d)