import numpy as np
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('M', type=int, help='First dimension of matrix A')
# parser.add_argument('K', type=int, help='Common dimension of matrices A and B')
# parser.add_argument('N', type=int, help='Second dimension of matrix B')
# args = parser.parse_args()

# A = np.random.rand(args.M, args.K).astype(np.float32)
# B = np.random.rand(args.K, args.N).astype(np.float32)

# C = np.dot(A, B)

# np.save('A.npy', A)
# np.save('B.npy', B)
# np.save('C.npy', C)

# A = np.arange(4 * 4).reshape([4, 4]).astype(np.float32)
# B = np.arange(4 * 8).reshape([4, 8]).astype(np.float32)
A = np.zeros([90, 90]).astype(np.float32)
B = np.zeros([90, 90]).astype(np.float32)
for i in range(0, 90):
  A[0, i] = B[0, i] = i
  A[i, 0] = B[i, 0] = i

C = np.dot(A, B)

np.save('A.npy', A)
np.save('B.npy', B)
np.save('C.npy', C)
