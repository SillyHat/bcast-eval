#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/bcast.h"
#include "../include/decision.h"

/*
 * Use this function to evaluate the execution time for various broadcast
 * algorithms with variable parameters. Feel free to change the parameters
 * in 'decision.h'
 */

void bcast_measure();

int main(int argc, char *argv[])
{
	//bcast_measure();
	decision_matrix_construct(1);
	return 0;
}

void bcast_measure()
{
	/*
	 * Initializing MPI routine. First getting world parameters,
	 * then creating the world group.
	 */

	int is_mpi;
	MPI_Initialized(&is_mpi);

	if (0 == is_mpi)
		MPI_Init(NULL, NULL);

	int world_size, world_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	MPI_Group world_group;
	MPI_Comm_group(MPI_COMM_WORLD, &world_group);

	/*
	 * A ranks array for splitting the MPI_COMM_WORLD thing.
	 */

	int ranks[world_size];

	for (int i = 0; i < world_size; ++i)
		ranks[i] = i;

	/*
	 * We don't want any worlds with less than 4 processes.
	 */

	if (world_size < 4) {
		fprintf(stderr, "Your world size is insufficient (%d). Cancelling...\n", world_size);
		exit(1);
	}

	MPI_Group new_group;

	for (unsigned int group_size = COMM_SIZE_MIN; group_size <= COMM_SIZE_MAX && group_size <= world_size; group_size += COMM_SIZE_STEP) {

		if (group_size < 4)
			continue;

		/*
		 * Creating a new communicator of desired size via group functions.
		 */

		MPI_Group_incl(world_group, group_size, ranks, &new_group);
		MPI_Comm new_comm;
		MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);

		int new_rank = -1, new_size = -1;

		/*
		 * If you belong to the newly created group, do the collectives.
		 */

		if (MPI_COMM_NULL != new_comm) {
			MPI_Comm_rank(new_comm, &new_rank);
			MPI_Comm_size(new_comm, &new_size);

			if (0 == new_rank) {
				printf("\nP = %u\n", group_size);
				printf("KBYTES  TIME_DFLT TIME_BNML TIME_SCRA TIME_SCDA\n");
			}

			MPI_Barrier(new_comm);

			for (unsigned long int buf_size = MESS_SIZE_MIN; buf_size <= MESS_SIZE_MAX; buf_size += MESS_SIZE_STEP) {

				if (buf_size < 1)
					continue;

				/*
				 * Broadcasting an array of chars (one char = one byte).
				 */

				char *buffer = (char *)malloc(sizeof(char) * buf_size);
				assert(buffer != NULL);

				if (0 == new_rank)
					memset(buffer, '1', buf_size);
				else
					memset(buffer, '0', buf_size);

				/*
				 * Here comes measuring an execution time of four broadcast algorithms.
				 * We have a default MPICH implementation and three custom functions,
				 * binomial tree, scatter + ring allgather, scatter + doubling allgather.
				 */

				double time_dflt = 0.0, time_bnml = 0.0, time_scra = 0.0, time_scda = 0.0;

				for (unsigned int i = 0; i < TRIALS_NO; ++i) {
					MPI_Barrier(new_comm);
					time_dflt -= MPI_Wtime();
					MPI_Bcast(buffer, buf_size, MPI_CHAR, 0, new_comm);
					MPI_Barrier(new_comm);
					time_dflt += MPI_Wtime();

					MPI_Barrier(new_comm);
					time_bnml -= MPI_Wtime();
					Bcast_binomial(buffer, buf_size, MPI_CHAR, 0, new_comm);
					MPI_Barrier(new_comm);
					time_bnml += MPI_Wtime();

					MPI_Barrier(new_comm);
					time_scra -= MPI_Wtime();
					Bcast_scatter_ring_allgather(buffer, buf_size, MPI_CHAR, 0, new_comm);
					MPI_Barrier(new_comm);
					time_scra += MPI_Wtime();

					MPI_Barrier(new_comm);
					time_scda -= MPI_Wtime();
					Bcast_scatter_doubling_allgather(buffer, buf_size, MPI_CHAR, 0, new_comm);
					MPI_Barrier(new_comm);
					time_scda += MPI_Wtime();
				}

				if (0 == new_rank) {
					printf("%6.0lf  %2.6lf  %2.6lf  %2.6lf  %2.6lf\n",
						(double)buf_size / 1024, time_dflt / TRIALS_NO, time_bnml / TRIALS_NO, time_scra / TRIALS_NO, time_scda / TRIALS_NO);
				}

				free(buffer);
			}

			/*
			 * Better dispose the current communicator here.
			 */

			MPI_Comm_free(&new_comm);
		}

		/*
		 * Free the current group, erect a barrier (so nothing bad happens).
		 */

		MPI_Barrier(MPI_COMM_WORLD);
	}

	/*
	 * And now we're done.
	 */

	MPI_Group_free(&new_group);
	MPI_Group_free(&world_group);
	MPI_Finalize();
}
