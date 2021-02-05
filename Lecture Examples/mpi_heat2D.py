#****************************************************************************
# FILE: mpi_heat2D.c
# OTHER FILES: draw_heat.c
# DESCRIPTIONS:
# HEAT2D Example - Parallelized C Version
# This example is based on a simplified two-dimensional heat
# equation domain decomposition.  The initial temperature is computed to be
# high in the middle of the domain and zero at the boundaries.  The
# boundaries are held at zero throughout the simulation.  During the
# time-stepping, an array containing two domains is used; these domains
# alternate between old data and new data.
#
# In this parallelized version, the grid is decomposed by the master
# process and then distributed by rows to the worker processes.  At each
# time step, worker processes must exchange border data with neighbors,
# because a grid point's current temperature depends upon it's previous
# time step value plus the values of the neighboring grid points.  Upon
# completion of all time steps, the worker processes return their results
# to the master process.
#
# Two data files are produced: an initial data set and a final data set.
# An X graphic of these two states displays after all calculations have
# completed.
# AUTHOR: Blaise Barney - adapted from D. Turner's serial C version. Converted
#   to MPI: George L. Gusciora (1/95)
# LAST REVISED: 06/12/13 Blaise Barney
# Converted to Python by SH (10/17)
#****************************************************************************/
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

#*****************************************************************************
#  function inidat
#*****************************************************************************/
def inidat(nx,ny,u):
    for ix in range(nx):
        for iy in range(ny):
            u[0,ix,iy] = ix * (nx - ix - 1) * iy * (ny - iy - 1)

#**************************************************************************
# function prtdat
#**************************************************************************/
def prtdat(nx,ny,u1,fnam):
    fp = open(fnam, 'w')
    for ix in range(nx):
        for iy in range(ny):
            print("{:8.1f}".format(u1[ix,iy]),end=' ',file=fp)
        fp.write("\n")
    fp.close()

#**************************************************************************
# function pltdat
#**************************************************************************/
def pltdat(u1,range,fnam):
    plt.imsave(fnam,u1,cmap='hot',vmin=range[0],vmax=range[1])
    plt.figure()
    plt.set_cmap('hot')
    plt.imshow(u1,vmin=range[0],vmax=range[1])
    plt.show()

#**************************************************************************
# function update
#****************************************************************************/
def update(start, end, ny, u1, u2):
    for ix in range(start,end+1):
        for iy in range(1,ny-1):
            u2[ix,iy] = u1[ix,iy] + Parms_Cx * (u1[ix+1,iy] + u1[ix-1,iy] - 2.0 * u1[ix,iy]) + Parms_Cy * (u1[ix,iy+1] + u1[ix,iy-1] - 2.0 * u1[ix,iy])


NXPROB     = 20         # x dimension of problem grid
NYPROB     = 20         # y dimension of problem grid
STEPS      = 100        # number of time steps
MAXWORKER  = 8          # maximum number of worker tasks
MINWORKER  = 3          # minimum number of worker tasks
BEGIN      = 1          # message tag
LTAG       = 2          # message tag
RTAG       = 3          # message tag
NONE       = 0          # indicates no neighbour
DONE       = 4          # message tag
MASTER     = 0          # taskid of first process

Parms_Cx = 0.1          # blend factor in heat equation
Parms_Cy = 0.1          # blend factor in heat equation

u = np.zeros((2,NXPROB,NYPROB))        # array for grid

# First, find out my taskid and how many tasks are running
comm = MPI.COMM_WORLD
taskid = comm.Get_rank()
numtasks = comm.Get_size()
numworkers = numtasks-1
    
#************************* master code *******************************/
if taskid == MASTER:
    # Check if numworkers is within range - quit if not
    if (numworkers > MAXWORKER) or (numworkers < MINWORKER):
        print("ERROR: the number of tasks must be between %d and %d." % (MINWORKER+1,MAXWORKER+1))
        print("Quitting...")
        comm.Abort()

    print("Starting mpi_heat2D with %d worker tasks." % numworkers)

    # Initialize grid
    print("Grid size: X= %d  Y= %d  Time steps= %d" % (NXPROB,NYPROB,STEPS))
    print("Initializing grid and writing initial.dat file...")
    inidat(NXPROB, NYPROB, u)
    prtdat(NXPROB, NYPROB, u[0], "initial.dat")
    plotrange = (u.min(),u.max())
    pltdat(u[0], plotrange, "initial.png")

    # Distribute work to workers.  Must first figure out how many rows to
    # send and what to do with extra rows.
    averow = NXPROB//numworkers
    extra = NXPROB%numworkers
    offset = 0

    for i in range(1,numworkers+1):
        rows = averow
        if i <= extra:
            rows+=1

        # Tell each worker who its neighbors are, since they must exchange
        # data with each other.
        if i == 1:
            left = NONE
        else:
            left = i - 1
        if i == numworkers:
            right = NONE
        else:
            right = i + 1

        # Now send startup information to each worker
        comm.send(offset, dest=i, tag=BEGIN)
        comm.send(rows, dest=i, tag=BEGIN)
        comm.send(left, dest=i, tag=BEGIN)
        comm.send(right, dest=i, tag=BEGIN)
        comm.Send(u[0,offset:offset+rows,:], dest=i, tag=BEGIN)
        print("Sent to task %d: rows= %d offset= %d" % (i,rows,offset),end=' ')
        print("left= %d right= %d" % (left,right))
        offset += rows

    # Now wait for results from all worker tasks
    for i in range(1,numworkers+1):
        offset = comm.recv(source=i, tag=DONE)
        rows = comm.recv(source=i, tag=DONE)
        comm.Recv([u[0,offset,:],rows*NYPROB,MPI.DOUBLE], source=i, tag=DONE)

    # Write final output, call X graph and finalize MPI
    print("Writing final.dat file and generating graph...")
    prtdat(NXPROB, NYPROB, u[0], "final.dat")
    pltdat(u[0],plotrange,"final.png")
    # End of master code
    
    
    
#************************* workers code **********************************/
elif taskid != MASTER:
    # Array is already initialized to zero - including the borders
    # Receive my offset, rows, neighbors and grid partition from master
    offset = comm.recv(source=MASTER, tag=BEGIN)
    rows = comm.recv(source=MASTER, tag=BEGIN)
    left = comm.recv(source=MASTER, tag=BEGIN)
    right = comm.recv(source=MASTER, tag=BEGIN)
    comm.Recv([u[0,offset,:],rows*NYPROB,MPI.DOUBLE], source=MASTER, tag=BEGIN)

    # Determine border elements.  Need to consider first and last columns.
    # Obviously, row 0 can't exchange with row 0-1.  Likewise, the last
    # row can't exchange with last+1.
    start=offset
    end=offset+rows-1
    if offset==0:
        start=1
    if (offset+rows)==NXPROB:
        end-=1
    print("task=%d  start=%d  end=%d"%(taskid,start,end))
        
    # Begin doing STEPS iterations.  Must communicate border rows with
    # neighbours.  If I have the first or last grid row, then I only need
    # to  communicate with one neighbour
    print("Task %d received work. Beginning time steps..." % taskid)
    iz = 0;
    for it in range(STEPS):
        if left != NONE:
            comm.Send([u[iz,offset,:],NYPROB,MPI.DOUBLE], dest=left, tag=RTAG)
            comm.Recv([u[iz,offset-1,:],NYPROB,MPI.DOUBLE], source=left, tag=LTAG)
        if right != NONE:
            comm.Send([u[iz,offset+rows-1,:],NYPROB,MPI.DOUBLE], dest=right, tag=LTAG)
            comm.Recv([u[iz,offset+rows,:],NYPROB,MPI.DOUBLE], source=right, tag=RTAG)
        # Now call update to update the value of grid points
        update(start,end,NYPROB,u[iz],u[1-iz]);
        iz = 1 - iz

    # Finally, send my portion of final results back to master
    comm.send(offset, dest=MASTER, tag=DONE)
    comm.send(rows, dest=MASTER, tag=DONE)
    comm.Send([u[iz,offset,:],rows*NYPROB,MPI.DOUBLE], dest=MASTER, tag=DONE)



