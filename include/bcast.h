#ifndef _BCAST_H
#define _BCAST_H

#include <mpi.h>
#include <stdlib.h>

void Bcast_binomial(void *data, int count, MPI_Datatype datatype, int root, MPI_Comm communicator);
void Bcast_scatter_ring_allgather(void *data, int count, MPI_Datatype datatype, int root, MPI_Comm communicator);
void Bcast_scatter_doubling_allgather(void *data, int count, MPI_Datatype datatype, int root, MPI_Comm communicator);

#endif
