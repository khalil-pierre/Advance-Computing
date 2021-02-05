#******************************************************************************
# FILE: mpi_array.py
# DESCRIPTION:
#   MPI Example - Array Assignment - Python mpi4py version.
#   This program demonstrates a simple data decomposition. The master task
#   first initializes an array and then distributes an equal portion that
#   array to the other tasks. After the other tasks receive their portion
#   of the array, they perform an addition operation to each array element.
#   They also maintain a sum for their portion of the array. The master task
#   does likewise with its portion of the array. As each of the non-master
#   tasks finish, they send their updated portion of the array to the master.
#   An MPI collective communication call is used to collect the sums
#   maintained by each task.  Finally, the master task displays selected
#   parts of the final array and the global sum of all array elements.
#
# AUTHOR: Blaise Barney 04/13/05
#   Adapted to Python / mpi4py by SH (27/10/18)
#   NOTE: the number of MPI tasks must be evenly divided by 4.
#
#****************************************************************************
from mpi4py import MPI
import numpy as np

ARRAYSIZE = 16000000  # total size of array to process
MASTER = 0            # taskid of master task

#**************************** functions ************************************

def update(myoffset, chunk, myid):
#
# Perform addition to each of my array elements and keep my sum
#
    mysum = 0
    for i in range(myoffset,myoffset + chunk):
        data[i] = data[i] + i * 1.0
        mysum += data[i]
    print("Task %d mysum = %e" % (myid,mysum))
    return mysum

#**************************** initializations ************************************
data = np.zeros(ARRAYSIZE,dtype=np.float64)
comm = MPI.COMM_WORLD
numtasks = comm.Get_size()

if (numtasks % 4 != 0):
    print("Number of tasks is",numtasks)
    print("Quitting. Number of MPI tasks must be divisible by 4.")
    comm.Abort()

taskid = comm.Get_rank()
print ("MPI task %d has started..." % taskid)
chunksize = int(ARRAYSIZE / numtasks)
tag2 = 1
tag1 = 2
#**************************** master task only ************************************
if (taskid == MASTER):
#
# Initialize the array
#
    for i in range(ARRAYSIZE):
        data[i] =  i * 1.0
    sum = np.sum(data)
    print("Initialized array sum = %e" % sum)
#
# Send each task its portion of the array - master keeps 1st part
#
    offset = chunksize
    for dest in range(1,numtasks):
        comm.send(offset, dest=dest, tag=tag1)
        comm.Send(data[offset:offset+chunksize], dest=dest, tag=tag2)
        print("Sent %d elements to task %d offset= %d" % (chunksize,dest,offset))
        offset += chunksize
#
# Master does its part of the work
#
    offset = 0
    mysum = update(offset, chunksize, taskid)
#
# Wait to receive results from each task
#
    for source in range(1,numtasks):
        offset = comm.recv(source=source, tag=tag1)
        comm.Recv([data,chunksize,MPI.DOUBLE], source=source, tag=tag2)
#
# Get final sum and print sample results
#
    sum = comm.reduce(mysum, None, root=MASTER)
    print("Sample results: ")
    offset = 0
    for i in range(numtasks):
        #        for j in range(5):
        print("Task %d: offset: %d data: %e  %e  %e  %e  %e" % (i,offset,data[offset],data[offset+1],data[offset+2],data[offset+3],data[offset+4]))
        offset = offset + chunksize
    print("*** Final sum= %e ***" % sum)
#
#**************************** worker tasks ************************************
if (taskid > MASTER):
#
# Receive my portion of array from the master task
#
    offset = comm.recv(source=MASTER, tag=tag1)
    comm.Recv([data,chunksize,MPI.DOUBLE], source=MASTER, tag=tag2)
    mysum = update(offset, chunksize, taskid)
#
# Send my results back to the master task
#
    comm.send(offset, dest=MASTER, tag=tag1)
    comm.Send(data[offset:offset+chunksize], dest=MASTER, tag=tag2)
    sum = comm.reduce(mysum, None, root=MASTER)
#**************************** end of main program ************************************

