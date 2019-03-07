#include "../include/bcast.h"

/*
 * The following algorithms were taken from MPICH 3.2.1 source code for academic purposes.
 * I do not own any credit for this code (except one bug fix).
 */

/*
 * A simple macros for finding a minimum of two values.
 */

#define min(a, b) __extension__ ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a < _b ? _a : _b; })

/*
 * A basic implementation of Jenkins' one-at-a-time hash function.
 * Used for validating the integrity of broadcasted message.
 */

unsigned int jenkins_hash(char *data, size_t nbytes)
{
	unsigned int hash, i;
	for (hash = i = 0; i < nbytes; ++i) {
		hash += data[i];
		hash += (hash << 10);
		hash ^= (hash >> 6);
	}
	hash += (hash << 3);
	hash ^= (hash >> 11);
	hash += (hash << 15);
	return hash;
}

/*
 * Here comes the utility function, a scatter operation for broadcast algorithms.
 * It is used in scatter + allgather algorithms and is implemented using a classic binomial tree algorithm.
 */

static void scatter_for_bcast(void *data, int count, MPI_Datatype datatype, int root, MPI_Comm communicator, int nbytes)
{
	MPI_Status status;
	int rank, comm_size, src, dst;
	int relative_rank, mask;
	int scatter_size, curr_size, recv_size = 0, send_size;

	MPI_Comm_size(communicator, &comm_size);
	MPI_Comm_rank(communicator, &rank);

	relative_rank = (rank >= root) ? rank - root : rank - root + comm_size;

	scatter_size = (nbytes + comm_size - 1) / comm_size;
	curr_size = (rank == root) ? nbytes : 0;

	mask = 0x1;
	while (mask < comm_size) {
		if (relative_rank & mask) {
			src = rank - mask;
			if (src < 0)
				src += comm_size;
			recv_size = nbytes - relative_rank * scatter_size;
			if (recv_size <= 0) {
				curr_size = 0;
			} else {
				MPI_Recv(((char *)data + relative_rank * scatter_size),
					recv_size, MPI_BYTE, src, 0, communicator, &status);
				MPI_Get_count(&status, MPI_BYTE, &curr_size);
			}
            		break;
		}
		mask <<= 1;
	}

	mask >>= 1;
	while (mask > 0) {
		if (relative_rank + mask < comm_size) {
			send_size = curr_size - scatter_size * mask;
			if (send_size > 0) {
				dst = rank + mask;
				if (dst >= comm_size)
					dst -= comm_size;
				MPI_Send(((char *)data + scatter_size * (relative_rank + mask)),
					send_size, MPI_BYTE, dst, 0, communicator);
				curr_size -= send_size;
			}
		}
		mask >>= 1;
	}
}

/*
 * A binomial tree broadcast algorithm. Good for short messages sent
 * among a small number of processes.
 */

void Bcast_binomial(void *data, int count, MPI_Datatype datatype, int root, MPI_Comm communicator)
{
	int rank, comm_size, src, dst;
	int relative_rank, mask;
	int nbytes = 0;
	int recvd_size;
	int type_size;
	MPI_Status status;

	MPI_Comm_size(communicator, &comm_size);
	MPI_Comm_rank(communicator, &rank);

	if (comm_size == 1)
		return;

	MPI_Type_size(datatype, &type_size);
	nbytes = type_size * count;

	if (nbytes == 0)
		return;

	relative_rank = (rank >= root) ? rank - root : rank - root + comm_size;

	mask = 0x1;
	while (mask < comm_size) {
		if (relative_rank & mask) {
			src = rank - mask;
			if (src < 0)
				src += comm_size;
			MPI_Recv(data, count, datatype, src, 0, communicator, &status);
			MPI_Get_count(&status, MPI_BYTE, &recvd_size);
			break;
		}
		mask <<= 1;
	}

	mask >>= 1;
	while (mask > 0) {
		if (relative_rank + mask < comm_size) {
			dst = rank + mask;
			if (dst >= comm_size)
				dst -= comm_size;
			MPI_Send(data, count, datatype, dst, 0, communicator);
		}
		mask >>= 1;
	}
}

/*
 * Broadcast algorithm based on a scatter followed by a ring allgather.
 * A ring algorithm may perform better than recursive doubling for long messages
 * and medium-sized non-power-of-two messages.
 */

void Bcast_scatter_ring_allgather(void *data, int count, MPI_Datatype datatype, int root, MPI_Comm communicator)
{
	int rank, comm_size;
	int scatter_size;
	int j, i;
	int nbytes, type_size;
	int left, right, jnext;
	int curr_size = 0;
	int recvd_size;
	MPI_Status status;

	MPI_Comm_size(communicator, &comm_size);
	MPI_Comm_rank(communicator, &rank);

	if (comm_size == 1)
		return;

	MPI_Type_size(datatype, &type_size);
	nbytes = type_size * count;

	if (nbytes == 0)
		return;

	scatter_size = (nbytes + comm_size - 1) / comm_size;

	scatter_for_bcast(data, count, datatype, root, communicator, nbytes);

	curr_size = min(scatter_size, nbytes - ((rank - root + comm_size) % comm_size) * scatter_size);

	if (curr_size < 0)
		curr_size = 0;

	left  = (comm_size + rank - 1) % comm_size;
	right = (rank + 1) % comm_size;
	j = rank;
	jnext = left;

	for (i = 1; i < comm_size; i++) {
		int left_count, right_count, left_disp, right_disp, rel_j, rel_jnext;

		rel_j = (j - root + comm_size) % comm_size;
		rel_jnext = (jnext - root + comm_size) % comm_size;
		left_count = min(scatter_size, (nbytes - rel_jnext * scatter_size));
		if (left_count < 0)
			left_count = 0;
		left_disp = rel_jnext * scatter_size;
		right_count = min(scatter_size, (nbytes - rel_j * scatter_size));
		if (right_count < 0)
			right_count = 0;
		right_disp = rel_j * scatter_size;

		MPI_Sendrecv((char *)data + right_disp, right_count, MPI_BYTE, right, 0,
			(char *)data + left_disp, left_count, MPI_BYTE, left, 0, communicator, &status);
		MPI_Get_count(&status, MPI_BYTE, &recvd_size);
		curr_size += recvd_size;
		j = jnext;
		jnext = (comm_size + jnext - 1) % comm_size;
	}
}

/*
 * Broadcast algorithm based on a scatter followed by an allgather step.
 * A recursive doubling allgather algorithm is good for medium-sized messages and
 * power-of-two number of processes. Non-power-of-two numbers of processes are also
 * taken care of.
 *
 * !!!!
 * I have fixed a critical error present in the original MPICH code.
 * A bug in the non-power-of-two section could lead to the negative values of
 * 'count' parameter for MPI_Recv function.
 * Good for me.
 */

void Bcast_scatter_doubling_allgather(void *data, int count, MPI_Datatype datatype, int root, MPI_Comm communicator)
{
	MPI_Status status;
	int rank, comm_size, dst;
	int relative_rank, mask;
	int scatter_size, curr_size, recv_size = 0;
	int j, k, i, tmp_mask;
	int type_size, nbytes = 0;
	int relative_dst, dst_tree_root, my_tree_root, send_offset;
	int recv_offset, tree_root, nprocs_completed, offset;

	MPI_Comm_size(communicator, &comm_size);
	MPI_Comm_rank(communicator, &rank);

	relative_rank = (rank >= root) ? rank - root : rank - root + comm_size;

	if (comm_size == 1)
		return;

	MPI_Type_size(datatype, &type_size);
	nbytes = type_size * count;
	if (nbytes == 0)
		return;

	scatter_size = (nbytes + comm_size - 1) / comm_size;

	scatter_for_bcast(data, count, datatype, root, communicator, nbytes);

	curr_size = min(scatter_size, (nbytes - (relative_rank * scatter_size)));

	if (curr_size < 0)
		curr_size = 0;

	mask = 0x1;
	i = 0;
	while (mask < comm_size) {
		relative_dst = relative_rank ^ mask;

		dst = (relative_dst + root) % comm_size;

		dst_tree_root = relative_dst >> i;
		dst_tree_root <<= i;

		my_tree_root = relative_rank >> i;
		my_tree_root <<= i;

		send_offset = my_tree_root * scatter_size;
		recv_offset = dst_tree_root * scatter_size;

		if (relative_dst < comm_size) {
			MPI_Sendrecv(((char *)data + send_offset), curr_size, MPI_BYTE, dst, 0,
				((char *)data + recv_offset), (nbytes - recv_offset < 0 ? 0 : nbytes - recv_offset),
				MPI_BYTE, dst, 0, communicator, &status);
			MPI_Get_count(&status, MPI_BYTE, &recv_size);
			curr_size += recv_size;
		}

		if (dst_tree_root + mask > comm_size) {
			nprocs_completed = comm_size - my_tree_root - mask;
			j = mask;
			k = 0;
			while (j) {
				j >>= 1;
				k++;
			}
			k--;

			offset = scatter_size * (my_tree_root + mask);
			tmp_mask = mask >> 1;

			while (tmp_mask) {
				relative_dst = relative_rank ^ tmp_mask;
				dst = (relative_dst + root) % comm_size;

				tree_root = relative_rank >> k;
				tree_root <<= k;

				if ((relative_dst > relative_rank) &&
					(relative_rank < tree_root + nprocs_completed) &&
					(relative_dst >= tree_root + nprocs_completed))
				{
					MPI_Send(((char *)data + offset), recv_size, MPI_BYTE, dst, 0, communicator);
				}
				else if ((relative_dst < relative_rank) &&
					(relative_dst < tree_root + nprocs_completed) &&
					(relative_rank >= tree_root + nprocs_completed))
				{
					MPI_Recv(((char *)data + offset), (nbytes - offset < 0 ? 0 : nbytes - offset),
						MPI_BYTE, dst, 0, communicator, &status);
					MPI_Get_count(&status, MPI_BYTE, &recv_size);
					curr_size += recv_size;
				}
				tmp_mask >>= 1;
				k--;
			}
		}
		mask <<= 1;
		i++;
	}
}
