from mpi4py import MPI
import numpy as np
from viztracer import log_sparse
np.set_printoptions(suppress=True)

def matmul(c: np.ndarray, a, b):
  c += a @ b

@log_sparse(stack_depth=3)
def cannon(A: np.ndarray, B: np.ndarray, C: np.ndarray, P, comm2d: MPI.Cartcomm):
  (M, K) = A.shape
  N = B.shape[1]
  (mTile, nTile, kTile) = (M // P, N // P, K // P)

  (j, i) = comm2d.Get_coords(rank)  # topology is row major
  # align data
  k = (i + j) % P
  a = np.ascontiguousarray(A[j * mTile:(j + 1) * mTile, k * kTile:(k + 1) * kTile].copy())
  b = np.ascontiguousarray(B[k * kTile:(k + 1) * kTile, i * nTile:(i + 1) * nTile].copy())

  # compute and shift
  c = np.empty([mTile, nTile], np.float32)
  for t in range(P):
    matmul(c.view(), a, b)
    if t == P - 1: continue
    # top right is (0,0)
    right, left = comm2d.Get_cart_rank([j, (i + 1) % P]), comm2d.Get_cart_rank([j, (i - 1) % P])
    comm2d.Sendrecv_replace(a, dest=left, source=right)
    top, down = comm2d.Get_cart_rank([(j - 1) % P, i]), comm2d.Get_cart_rank([(j + 1) % P, i])
    comm2d.Sendrecv_replace(b, dest=top, source=down)

  # compare result
  ref = C[j * mTile:(j + 1) * mTile, i * nTile:(i + 1) * nTile]
  assert np.allclose(c, ref)


if __name__ == '__main__':
  A, B, C = np.load('A.npy', 'r'), np.load('B.npy', 'r'), np.load('C.npy', 'r')
  P = 3
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  comm2d = comm.Create_cart([P, P], [False, False])
  cannon(A, B, C, P, comm2d)
