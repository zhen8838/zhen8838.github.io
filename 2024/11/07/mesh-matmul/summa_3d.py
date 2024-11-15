import sys
from mpi4py import MPI
from viztracer import log_sparse
import numpy as np
np.set_printoptions(suppress=True)


@log_sparse(stack_depth=3)
def summa_3d(A: np.ndarray, B: np.ndarray, C: np.ndarray, p, comm3d: MPI.Cartcomm):
  x_comm = comm3d.Sub([False, False, True])
  y_comm = comm3d.Sub([False, True, False])
  z_comm = comm3d.Sub([True, False, False])

  (M, K) = A.shape
  N = B.shape[1]
  (MTile, NTile, KTile) = (M // p, N // p, K // p)
  (mTile, nTile, kTile) = (MTile // p, NTile // p, KTile // p)
  (l, j, i) = comm3d.Get_coords(rank)  # topology is row major
  # align data
  a = np.ascontiguousarray(A[i * MTile:(i + 1) * MTile, l * KTile:(l + 1) * KTile]
                           [:, j * kTile: (j + 1) * kTile].copy())
  b = np.ascontiguousarray(B[l * KTile:(l + 1) * KTile, j * NTile:(j + 1)
                           * NTile][:, i * nTile: (i + 1) * nTile].copy())

  # compute and passing data
  c = np.zeros([MTile, nTile], np.float32)
  Atemp = np.empty([p, MTile, kTile], np.float32)
  Btemp = np.empty([p, KTile, nTile], np.float32)
  y_comm.Allgather(a, Atemp)
  x_comm.Allgather(b, Btemp)
  Atemp = Atemp.transpose([1, 0, 2]).reshape(MTile, KTile)
  Btemp = Btemp.transpose([1, 0, 2]).reshape(KTile, NTile)
  Dl = np.dot(Atemp, Btemp)  # [MTile, NTile]
  Dr = np.empty([p, MTile, nTile], np.float32)
  # Note that mpi4py will send data sequentially. So, we want to resplit on N. We had to split it first.
  Dsend = np.ascontiguousarray(np.stack(np.split(Dl, p, axis=-1)))
  z_comm.Alltoall(Dsend, Dr)
  c += np.sum(Dr, axis=0, keepdims=False)

  # compare result
  ref = C[i * MTile:(i + 1) * MTile, j * NTile:(j + 1) * NTile][:, l * nTile:(l + 1) * nTile]
  assert np.allclose(c, ref)


if __name__ == '__main__':
  A, B, C = np.load('A.npy', 'r'), np.load('B.npy', 'r'), np.load('C.npy', 'r')
  p = 2
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  comm3d = comm.Create_cart([p, p, p], [False, False, False])
  summa_3d(A, B, C, p, comm3d)
