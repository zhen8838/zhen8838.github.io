from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
m = 2
n = 4
k = 2
p = 2

senddata = np.ones([m, n], dtype=float)
for i in range(n):
  senddata[:, i] = (rank * p * n) + i
# mpi4py 发送的时候直接读的buffer, 只是transpose时buffer没有被修改, 并且是把buffer直接 flatten 切分的, 为了分离n维度, 这里必须reshape再transpose改变数据顺序.
# senddata = senddata
# senddata = rank * np.ones([m, n], dtype=int)
# task[j] send[i] => task[i] recv[j]
print(f"rank {rank} sendding {senddata}")
senddata = np.ascontiguousarray(np.stack(np.split(senddata, size, axis=-1)))
# senddata = np.transpose(senddata.reshape(m, n // size, size), [2, 0, 1]).copy()
recvdata = np.empty([size, m, n // size], dtype=float)
comm.Alltoall(senddata, recvdata)

print("process %s sending %s receiving %s \n" % (rank, senddata, recvdata))
