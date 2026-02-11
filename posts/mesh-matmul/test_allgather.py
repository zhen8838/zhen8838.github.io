from mpi4py import MPI
import numpy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
m = 3
n = 8
k = 4
senddata = rank * numpy.ones([m, k // size], dtype=int)
recvdata = numpy.empty([size, m, k // size], dtype=int)
comm.Allgather(senddata, recvdata)

print("process %s sending %s receiving %s transpose %s \n" %
      (rank, senddata, recvdata, recvdata.transpose(1, 0, 2).reshape([m, k])))
