#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include "auxiliar.h"

typedef struct {
    int    index_plane;
    double x;
    double y;
    double vx;
    double vy;
} MinPlaneToSend;

void read_planes_mpi(const char* filename, PlaneList* planes, int* N, int* M, double* x_max, double* y_max, int rank, int size, int* tile_displacements) {
    //Same as sequential but with little changes:

    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return;
    }

    char line[MAX_LINE_LENGTH];
    int num_planes = 0;

    fgets(line, sizeof(line), file);
    fgets(line, sizeof(line), file);
    sscanf(line, "# Map: %lf, %lf : %d %d", x_max, y_max, N, M);
    fgets(line, sizeof(line), file);
    sscanf(line, "# Number of Planes: %d", &num_planes);
    fgets(line, sizeof(line), file);

    //Domain decomposition!!
    for (int i = 0; i <= size; i++) {
        tile_displacements[i] = i * ((*N) * (*M) / size);
    }

    int index = 0;
    while (fgets(line, sizeof(line), file)) {
        int idx;
        double x, y, vx, vy;
        if (sscanf(line, "%d %lf %lf %lf %lf", &idx, &x, &y, &vx, &vy) == 5) {
            index++;
            int index_i = get_index_i(x, *x_max, *N);
            int index_j = get_index_j(y, *y_max, *M);
            int index_map = get_index(index_i, index_j, *N, *M);
            // Each process only inserts planes that fall into its assigned tile range.
            // This conditional ensures a process only processes its share of the map.
            if (index_map >= tile_displacements[rank] && index_map < tile_displacements[rank + 1]) {
                    insert_plane(planes, idx, index_map, rank, x, y, vx, vy);
                    index++;// Only count planes actually inserted by this rank
                
            }
        }
    }
    fclose(file);
    printf("Total planes read: %d\n", index);
    assert(num_planes == index);
}

void communicate_planes_send(PlaneList* list, int N, int M, double x_max, double y_max, int rank, int size, int* tile_displacements) {
    // STEP 1: Count how many planes must be sent to each process
    // For each plane, compute its new destination rank based on its coordinates.
    // This allows us to prepare communication buffers sized correctly.
    int *send_counts = calloc(size, sizeof(int));
    for (PlaneNode *node = list->head; node; node = node->next) {
        int i = get_index_i(node->x, x_max, N);
        int j = get_index_j(node->y, y_max, M);
        int dest_rank = get_rank_from_indices(i, j, N, M, tile_displacements, size);
        if (dest_rank != rank) send_counts[dest_rank]++;
    }

    // STEP 2: Allocate one buffer per destination to pack outgoing plane data
    // Each buffer will store all planes we need to send to that particular rank
    double **send_buffers = malloc(size * sizeof(double*));
    int *buffer_offsets = calloc(size, sizeof(int));
    for (int r = 0; r < size; r++) {
        send_buffers[r] = malloc(send_counts[r] * 6 * sizeof(double));  // each plane has 6 doubles
    }

    // STEP 3: Pack planes into buffers and remove them from the local list
    // We prepare each plane for sending by writing it to its corresponding buffer
    PlaneNode *node = list->head;
    while (node) {
        PlaneNode *next_node = node->next;
        int i = get_index_i(node->x, x_max, N);
        int j = get_index_j(node->y, y_max, M);
        int dest_rank = get_rank_from_indices(i, j, N, M, tile_displacements, size);
        if (dest_rank != rank) {
            int offset = buffer_offsets[dest_rank]++;
            double *buf = send_buffers[dest_rank] + 6 * offset;
            buf[0] = (double)node->index_plane;
            buf[1] = node->x;
            buf[2] = node->y;
            buf[3] = node->vx;
            buf[4] = node->vy;
            buf[5] = (double)node->index_map;
            remove_plane(list, node);
        }
        node = next_node;
    }

    // STEP 4: Exchange send/receive counts to coordinate message sizes
    // Avoid deadlocks by respecting rank ordering: lower ranks receive first
    int *recv_counts = calloc(size, sizeof(int));
    MPI_Request reqs[2];
    for (int r = 0; r < size; r++) {
        if (r == rank) continue;
        if (r < rank) {
            MPI_Recv(&recv_counts[r], 1, MPI_INT, r, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Isend(&send_counts[r], 1, MPI_INT, r, 10, MPI_COMM_WORLD, &reqs[0]);
        } else {
            MPI_Isend(&send_counts[r], 1, MPI_INT, r, 10, MPI_COMM_WORLD, &reqs[0]);
            MPI_Recv(&recv_counts[r], 1, MPI_INT, r, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (r < rank)
            MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);
    }

    // STEP 5: Allocate memory to receive incoming planes
    double **recv_buffers = malloc(size * sizeof(double*));
    for (int r = 0; r < size; r++) {
        recv_buffers[r] = (r == rank ? NULL : malloc(recv_counts[r] * 6 * sizeof(double)));
    }

    // STEP 6: Perform actual exchange of plane data
    // Use same rank-ordering strategy as above to avoid deadlocks
    for (int r = 0; r < size; r++) {
        if (r == rank) continue;
        if (r < rank) {
            MPI_Recv(recv_buffers[r], recv_counts[r] * 6, MPI_DOUBLE, r, 20, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Isend(send_buffers[r], send_counts[r] * 6, MPI_DOUBLE, r, 20, MPI_COMM_WORLD, &reqs[1]);
        } else {
            MPI_Isend(send_buffers[r], send_counts[r] * 6, MPI_DOUBLE, r, 20, MPI_COMM_WORLD, &reqs[1]);
            MPI_Recv(recv_buffers[r], recv_counts[r] * 6, MPI_DOUBLE, r, 20, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (r < rank)
            MPI_Wait(&reqs[1], MPI_STATUS_IGNORE);
    }

    // STEP 7: Unpack received planes and insert them into local list
    for (int r = 0; r < size; r++) {
        if (r == rank) continue;
        for (int i = 0; i < recv_counts[r]; i++) {
            double *buf = recv_buffers[r] + 6 * i;
            int plane_id = (int) buf[0];
            double x = buf[1];
            double y = buf[2];
            double vx = buf[3];
            double vy = buf[4];
            int grid_index = (int) buf[5];
            insert_plane(list, plane_id, grid_index, rank, x, y, vx, vy);
        }
    }

    // STEP 8: Free all dynamically allocated memory
    for (int r = 0; r < size; r++) {
        free(send_buffers[r]);
        if (r != rank) free(recv_buffers[r]);
    }
    free(send_buffers);
    free(recv_buffers);
    free(send_counts);
    free(recv_counts);
    free(buffer_offsets);
}

void communicate_planes_alltoall(PlaneList* list, int N, int M, double x_max, double y_max, int rank, int size, int* tile_displacements) {
    // Step 1: Count how many planes to send to each rank
    int* send_counts = calloc(size, sizeof(int));
    MinPlaneToSend** plane_buckets = calloc(size, sizeof(MinPlaneToSend*));
    int* plane_bucket_sizes = calloc(size, sizeof(int));

    PlaneNode* current = list->head;
    while (current != NULL) {
        PlaneNode* next = current->next;
        int i = get_index_i(current->x, x_max, N);
        int j = get_index_j(current->y, y_max, M);
        int index_map = get_index(i, j, N, M);
        int new_rank = get_rank_from_index(index_map, tile_displacements, size);

        if (new_rank != rank) {
            MinPlaneToSend mp = {
                .index_plane = current->index_plane,
                .x = current->x,
                .y = current->y,
                .vx = current->vx,
                .vy = current->vy
            };
            plane_bucket_sizes[new_rank]++;
            plane_buckets[new_rank] = realloc(plane_buckets[new_rank], plane_bucket_sizes[new_rank] * sizeof(MinPlaneToSend));
            plane_buckets[new_rank][plane_bucket_sizes[new_rank] - 1] = mp;
            remove_plane(list, current);
        }

        current = next;
    }

    for (int i = 0; i < size; i++) {
        send_counts[i] = plane_bucket_sizes[i];
    }

    // Step 2: Exchange counts
    int* recv_counts = calloc(size, sizeof(int));
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);

    // Step 3: Prepare send/recv buffers and displacements
    int total_send = 0, total_recv = 0;
    int* sdispls = calloc(size, sizeof(int));
    int* rdispls = calloc(size, sizeof(int));
    for (int i = 0; i < size; i++) {
        sdispls[i] = total_send;
        total_send += send_counts[i];
        rdispls[i] = total_recv;
        total_recv += recv_counts[i];
    }

    MinPlaneToSend* sendbuf = malloc(total_send * sizeof(MinPlaneToSend));
    MinPlaneToSend* recvbuf = malloc(total_recv * sizeof(MinPlaneToSend));

    for (int i = 0, offset = 0; i < size; i++) {
        if (send_counts[i] > 0) {
            memcpy(&sendbuf[offset], plane_buckets[i], send_counts[i] * sizeof(MinPlaneToSend));
            offset += send_counts[i];
        }
        free(plane_buckets[i]);
    }

    // Step 4: All-to-all data exchange
    MPI_Datatype MPI_MinPlane;
    MPI_Type_contiguous(5, MPI_DOUBLE, &MPI_MinPlane);  // Incorrect. Must define with offsets.
    MPI_Type_commit(&MPI_MinPlane);

    MPI_Alltoallv(sendbuf, send_counts, sdispls, MPI_MinPlane,
                  recvbuf, recv_counts, rdispls, MPI_MinPlane, MPI_COMM_WORLD);

    // Step 5: Reinsert received planes
    for (int i = 0; i < total_recv; i++) {
        MinPlaneToSend* p = &recvbuf[i];
        int index_i = get_index_i(p->x, x_max, N);
        int index_j = get_index_j(p->y, y_max, M);
        int index_map = get_index(index_i, index_j, N, M);
        insert_plane(list, p->index_plane, index_map, rank, p->x, p->y, p->vx, p->vy);
    }

    free(send_counts);
    free(recv_counts);
    free(sdispls);
    free(rdispls);
    free(plane_buckets);
    free(plane_bucket_sizes);
    free(sendbuf);
    free(recvbuf);
    MPI_Type_free(&MPI_MinPlane);
}

void communicate_planes_struct_mpi(PlaneList* list,int N, int M,double x_max, double y_max,int rank, int size,int* tile_displacements) {
    // Step 1: Create custom MPI datatype
    MPI_Datatype MPI_MinPlane;
    int lengths[5] = {1, 1, 1, 1, 1};
    MPI_Aint displs[5];
    MPI_Aint base;
    MinPlaneToSend dummy;
    MPI_Get_address(&dummy, &base);
    MPI_Get_address(&dummy.index_plane, &displs[0]);
    MPI_Get_address(&dummy.x, &displs[1]);
    MPI_Get_address(&dummy.y, &displs[2]);
    MPI_Get_address(&dummy.vx, &displs[3]);
    MPI_Get_address(&dummy.vy, &displs[4]);
    for (int i = 0; i < 5; i++) displs[i] -= base;

    MPI_Datatype types[5] = {MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    MPI_Type_create_struct(5, lengths, displs, types, &MPI_MinPlane);
    MPI_Type_commit(&MPI_MinPlane);

    // Step 2: Classify planes to send
    int* send_counts = calloc(size, sizeof(int));
    MinPlaneToSend** send_lists = calloc(size, sizeof(MinPlaneToSend*));

    PlaneNode* current = list->head;
    while (current != NULL) {
        PlaneNode* next = current->next;
        int i = get_index_i(current->x, x_max, N);
        int j = get_index_j(current->y, y_max, M);
        int index_map = get_index(i, j, N, M);
        int target = get_rank_from_index(index_map, tile_displacements, size);

        if (target != rank) {
            send_lists[target] = realloc(send_lists[target], (send_counts[target] + 1) * sizeof(MinPlaneToSend));
            send_lists[target][send_counts[target]] = (MinPlaneToSend){
                .index_plane = current->index_plane,
                .x = current->x,
                .y = current->y,
                .vx = current->vx,
                .vy = current->vy
            };
            send_counts[target]++;
            remove_plane(list, current);
        }
        current = next;
    }

    // Step 3: Alltoall exchange of counts
    int* recv_counts = calloc(size, sizeof(int));
    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);

    // Step 4: Build buffers and displacements
    int* sdispls = calloc(size, sizeof(int));
    int* rdispls = calloc(size, sizeof(int));
    int total_send = 0, total_recv = 0;
    for (int i = 0; i < size; i++) {
        sdispls[i] = total_send;
        rdispls[i] = total_recv;
        total_send += send_counts[i];
        total_recv += recv_counts[i];
    }

    MinPlaneToSend* sendbuf = malloc(total_send * sizeof(MinPlaneToSend));
    MinPlaneToSend* recvbuf = malloc(total_recv * sizeof(MinPlaneToSend));

    for (int i = 0, offset = 0; i < size; i++) {
        if (send_counts[i] > 0) {
            memcpy(&sendbuf[offset], send_lists[i], send_counts[i] * sizeof(MinPlaneToSend));
            offset += send_counts[i];
        }
        free(send_lists[i]);
    }

    // Step 5: Alltoallv communication
    MPI_Alltoallv(sendbuf, send_counts, sdispls, MPI_MinPlane,
                  recvbuf, recv_counts, rdispls, MPI_MinPlane, MPI_COMM_WORLD);

    // Step 6: Reinsert received planes
    for (int i = 0; i < total_recv; i++) {
        MinPlaneToSend* p = &recvbuf[i];
        int index_i = get_index_i(p->x, x_max, N);
        int index_j = get_index_j(p->y, y_max, M);
        int index_map = get_index(index_i, index_j, N, M);
        insert_plane(list, p->index_plane, index_map, rank, p->x, p->y, p->vx, p->vy);
    }

    // Cleanup
    free(send_counts);
    free(recv_counts);
    free(sdispls);
    free(rdispls);
    free(send_lists);
    free(sendbuf);
    free(recvbuf);
    MPI_Type_free(&MPI_MinPlane);
}

int main(int argc, char **argv) {
    int debug = 1;
    int N = 0, M = 0;
    double x_max = 0.0, y_max = 0.0;
    int max_steps;
    char* input_file;
    int check;
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int tile_displacements[size+1];
    int mode = 0;

    if (argc == 5) {
        input_file = argv[1];
        max_steps = atoi(argv[2]);
        mode = atoi(argv[3]);
        check = atoi(argv[4]);

        if (max_steps <= 0 || mode < 0 || mode > 2 || check < 0 || check > 1) {
            if (rank == 0) {
                fprintf(stderr, "Usage: %s <filename> <max_steps> <mode (0-2)> <check (0-1)>\n", argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    } else {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <filename> <max_steps> <mode> <check>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    PlaneList owning_planes = {NULL, NULL};
    read_planes_mpi(input_file, &owning_planes, &N, &M, &x_max, &y_max, rank, size, tile_displacements);
    PlaneList owning_planes_t0 = copy_plane_list(&owning_planes);

    double time_sim = 0., time_comm = 0, time_total = 0;
    double start_time = MPI_Wtime();

    for (int step = 1; step <= max_steps; step++) {
        double start = MPI_Wtime();
        PlaneNode* current = owning_planes.head;
        while (current != NULL) {
            current->x += current->vx;
            current->y += current->vy;
            current = current->next;
        }
        filter_planes(&owning_planes, x_max, y_max);
        time_sim += MPI_Wtime() - start;

        start = MPI_Wtime();
        if (mode == 0)
            communicate_planes_send(&owning_planes, N, M, x_max, y_max, rank, size, tile_displacements);
        else if (mode == 1)
            communicate_planes_alltoall(&owning_planes, N, M, x_max, y_max, rank, size, tile_displacements);
        else
            communicate_planes_struct_mpi(&owning_planes, N, M, x_max, y_max, rank, size, tile_displacements);
        time_comm += MPI_Wtime() - start;
    }

    time_total = MPI_Wtime() - start_time;

    if (rank == 0) {
        printf("Flight controller simulation: #input %s mode: %d size: %d\n", input_file, mode, size);
        printf("Time simulation:     %.2fs\n", time_sim);
        printf("Time communication:  %.2fs\n", time_comm);
        printf("Time total:          %.2fs\n", time_total);
    }

    if (check == 1)
        check_planes_mpi(&owning_planes_t0, &owning_planes, N, M, x_max, y_max, max_steps, tile_displacements, size, debug);

    MPI_Finalize();
    return 0;
}
