#include "../include/bcast.h"
#include "../include/decision.h"

#define BNML 0
#define SCDA 1
#define SCRA 2

unsigned long int matr_dim_i, matr_dim_j;
const char* algo_names[3] = {"bnml", "scda", "scra"};

static void decision_matrix_init();
static void decision_matrix_output();

/*
 * This routine makes a full decision map (matrix) for given communicator and message size ranges.
 */

void decision_matrix_construct(int verbose)
{
	/*
	 * Check if MPI_Init was called before.
	 */

	int is_mpi;
	MPI_Initialized(&is_mpi);

	if (0 == is_mpi) {
		MPI_Init(NULL, NULL);
	}

	int world_size, world_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	MPI_Group world_group;
	MPI_Comm_group(MPI_COMM_WORLD, &world_group);

	/*
	 * Terminate if we don't have the needed amount of processes.
	 */

	if (world_size < COMM_SIZE_MAX) {
		fprintf(stderr, "Can't construct a full decision matrix for this world size.\n");
		exit(1);
	}

	int ranks[world_size];

	for (int i = 0; i < world_size; ++i)
		ranks[i] = i;

	if (0 == world_rank)
		decision_matrix_init();

	int matrix_i = 0, matrix_j = 0;

	MPI_Group new_group;

	for (unsigned int group_size = COMM_SIZE_MIN; group_size <= COMM_SIZE_MAX && group_size <= world_size; group_size += COMM_SIZE_STEP) {

		if (group_size < 4)
			continue;

		/*
		 * Create a new communicator via group routines.
		 */

		MPI_Group_incl(world_group, group_size, ranks, &new_group);
		MPI_Comm new_comm;
		MPI_Comm_create_group(MPI_COMM_WORLD, new_group, 0, &new_comm);

		int new_rank = -1, new_size = -1;

		/*
		 * Do the following if you belong to a newly created communicator.
		 */

		if (MPI_COMM_NULL != new_comm) {

			MPI_Comm_rank(new_comm, &new_rank);
			MPI_Comm_size(new_comm, &new_size);

			matrix_j = 0;

			for (unsigned long int buf_size = MESS_SIZE_MIN; buf_size <= MESS_SIZE_MAX; buf_size += MESS_SIZE_STEP) {

				if (buf_size < 1)
					continue;

				/*
				 * We are broadcasting an array of chars.
				 */

				char *buffer = (char *)malloc(sizeof(char) * buf_size);
				assert(buffer != NULL);

				if (0 == new_rank)
					memset(buffer, '1', buf_size);
				else
					memset(buffer, '0', buf_size);

				/*
				 * Here we measure an execution time of three basic broadcast algorithms.
				 * Binomial tree, scatter + doubling allgather and scatter + ring allgather are being evaluated.
				 */

				double time[3] = {0.0, 0.0, 0.0};

				for (unsigned int i = 0; i < TRIALS_NO; ++i) {
					MPI_Barrier(new_comm);
					time[BNML] -= MPI_Wtime();
					Bcast_binomial(buffer, buf_size, MPI_CHAR, 0, new_comm);
					MPI_Barrier(new_comm);
					time[BNML] += MPI_Wtime();

					MPI_Barrier(new_comm);
					time[SCDA] -= MPI_Wtime();
					Bcast_scatter_doubling_allgather(buffer, buf_size, MPI_CHAR, 0, new_comm);
					MPI_Barrier(new_comm);
					time[SCDA] += MPI_Wtime();

					MPI_Barrier(new_comm);
					time[SCRA] -= MPI_Wtime();
					Bcast_scatter_ring_allgather(buffer, buf_size, MPI_CHAR, 0, new_comm);
					MPI_Barrier(new_comm);
					time[SCRA] += MPI_Wtime();
				}

				MPI_Barrier(new_comm);

				if (0 == new_rank) {
					unsigned char algo_min = BNML;
					double time_min = time[BNML] / TRIALS_NO;

					for (unsigned int i = 1; i < 3; ++i) {
						if (time_min > time[i] / TRIALS_NO) {
							algo_min = i;
							time_min = time[i] / TRIALS_NO;
						}
					}

					decision_matrix[matrix_i][matrix_j] = algo_min;

					if (0 != verbose)
						printf("P = %3d, BUF = %6.0lf KB. ALGO: %s [%2.6lf s]\n", new_size, (double)buf_size / 1024, algo_names[algo_min], time_min);
				}

				++matrix_j;
				free(buffer);
			}

			++matrix_i;
			MPI_Comm_free(&new_comm);
		}

		MPI_Barrier(MPI_COMM_WORLD);
	}

	if (0 == world_rank)
		decision_matrix_output();

	MPI_Group_free(&new_group);
	MPI_Group_free(&world_group);
	MPI_Finalize();

	for (unsigned int i = 0; i < matr_dim_i; ++i)
		free(decision_matrix[i]);
	free(decision_matrix);
}

static void decision_matrix_init()
{
	matr_dim_i = ceil(((double)(COMM_SIZE_MAX - COMM_SIZE_MIN + 1) / COMM_SIZE_STEP));
	matr_dim_j = ceil(((double)(MESS_SIZE_MAX - MESS_SIZE_MIN + 1) / MESS_SIZE_STEP));

	decision_matrix = (unsigned char **)malloc(sizeof(unsigned char *) * matr_dim_i);
	if (decision_matrix == NULL) {
		fprintf(stderr, "Error allocating memory for the matrix.\n");
		exit(1);
	}

	for (unsigned int i = 0; i < matr_dim_i; ++i) {
		decision_matrix[i] = (unsigned char *)malloc(sizeof(unsigned char) * matr_dim_j);
		if (decision_matrix[i] == NULL) {
			fprintf(stderr, "Error allocating memory for the matrix.\n");
			for (unsigned int j = 0; j < i; ++j)
				free(decision_matrix[j]);
			free(decision_matrix);
			exit(1);
		}
	}

	for (unsigned int i = 0; i < matr_dim_i; ++i)
		for (unsigned int j = 0; j < matr_dim_j; ++j)
			decision_matrix[i][j] = -1;
}

static void decision_matrix_output()
{
	FILE *fp;
	fp = fopen("decmatr.txt", "w");

	if (fp == NULL) {
		fprintf(stderr, "Error opening a file for writing.\n");
		return;
	}

	unsigned int actual_comm_min = COMM_SIZE_MIN, actual_mess_min = MESS_SIZE_MIN;

	while (actual_comm_min < 4)
		actual_comm_min += COMM_SIZE_STEP;

	while (actual_mess_min < 1)
		actual_mess_min += MESS_SIZE_STEP;

	fprintf(fp, "%u %u %u\n", actual_comm_min, COMM_SIZE_MAX, COMM_SIZE_STEP);
	fprintf(fp, "%u %u %u\n", actual_mess_min, MESS_SIZE_MAX, MESS_SIZE_STEP);

	for (unsigned int i = 0; i < matr_dim_i; ++i) {
		for (unsigned int j = 0; j < matr_dim_j && decision_matrix[i][j] != -1; ++j)
			fprintf(fp, "%hhu ", decision_matrix[i][j]);
		fprintf(fp, "\n");
	}
	fclose(fp);
}
