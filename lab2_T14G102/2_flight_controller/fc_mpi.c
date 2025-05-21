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

void communicate_planes_send(PlaneList* list,
                             int N, int M,
                             double x_max, double y_max,
                             int rank, int size,
                             int* tile_displacements)
{
    // STEP 1: Compute how many planes to send to each rank
    int* send_counts_per_rank = calloc(size, sizeof(int));
    for (PlaneNode* node = list->head; node; node = node->next) {
        int idx_i = get_index_i(node->x, x_max, N);
        int idx_j = get_index_j(node->y, y_max, M);
        int dest_rank = get_rank_from_indices(idx_i, idx_j, N, M, tile_displacements, size);
        if (dest_rank != rank) {
            send_counts_per_rank[dest_rank]++;
        }
    }

    // STEP 2: Allocate per-rank send buffers and track offsets
    double** plane_send_buffers = malloc(size * sizeof(double*));
    int* send_buffer_offsets = calloc(size, sizeof(int));
    for (int r = 0; r < size; ++r) {
        plane_send_buffers[r] = malloc(send_counts_per_rank[r] * 6 * sizeof(double));
    }

    // STEP 3: Pack local planes into send buffers and remove them from list
    for (PlaneNode* node = list->head, *next_node; node; node = next_node) {
        next_node = node->next;
        int idx_i = get_index_i(node->x, x_max, N);
        int idx_j = get_index_j(node->y, y_max, M);
        int dest_rank = get_rank_from_indices(idx_i, idx_j, N, M, tile_displacements, size);
        if (dest_rank != rank) {
            int offset = send_buffer_offsets[dest_rank]++;
            double* buf = plane_send_buffers[dest_rank] + offset * 6;
            buf[0] = (double)node->index_plane;
            buf[1] = node->x;
            buf[2] = node->y;
            buf[3] = node->vx;
            buf[4] = node->vy;
            buf[5] = (double)get_index(idx_i, idx_j, N, M);
            remove_plane(list, node);
        }
    }

    // STEP 4: Non-blocking exchange of send counts
    int* recv_counts_per_rank = calloc(size, sizeof(int));
    MPI_Request* count_requests = malloc(2 * size * sizeof(MPI_Request));
    int req_idx = 0;
    for (int r = 0; r < size; ++r) {
        if (r == rank) continue;
        MPI_Irecv(&recv_counts_per_rank[r], 1, MPI_INT, r, 0, MPI_COMM_WORLD, &count_requests[req_idx++]);
        MPI_Isend(&send_counts_per_rank[r], 1, MPI_INT, r, 0, MPI_COMM_WORLD, &count_requests[req_idx++]);
    }
    MPI_Waitall(req_idx, count_requests, MPI_STATUSES_IGNORE);

    // STEP 5: Allocate per-rank receive buffers
    double** plane_recv_buffers = malloc(size * sizeof(double*));
    for (int r = 0; r < size; ++r) {
        if (r == rank) plane_recv_buffers[r] = NULL;
        else plane_recv_buffers[r] = malloc(recv_counts_per_rank[r] * 6 * sizeof(double));
    }

    // STEP 6: Non-blocking exchange of plane data
    MPI_Request* data_requests = malloc(2 * size * sizeof(MPI_Request));
    req_idx = 0;
    for (int r = 0; r < size; ++r) {
        if (r == rank) continue;
        MPI_Irecv(plane_recv_buffers[r], recv_counts_per_rank[r] * 6, MPI_DOUBLE,
                  r, 1, MPI_COMM_WORLD, &data_requests[req_idx++]);
        MPI_Isend(plane_send_buffers[r], send_counts_per_rank[r] * 6, MPI_DOUBLE,
                  r, 1, MPI_COMM_WORLD, &data_requests[req_idx++]);
    }
    MPI_Waitall(req_idx, data_requests, MPI_STATUSES_IGNORE);

    // STEP 7: Unpack received planes into the local list
    for (int r = 0; r < size; ++r) {
        if (r == rank) continue;
        for (int i = 0; i < recv_counts_per_rank[r]; ++i) {
            double* buf = plane_recv_buffers[r] + i * 6;
            int  plane_id   = (int)buf[0];
            double x        = buf[1];
            double y        = buf[2];
            double vx       = buf[3];
            double vy       = buf[4];
            int    grid_idx = (int)buf[5];
            insert_plane(list, plane_id, grid_idx, rank, x, y, vx, vy);
        }
    }

    // STEP 8: Clean up all allocated memory and requests
    for (int r = 0; r < size; ++r) {
        free(plane_send_buffers[r]);
        if (plane_recv_buffers[r]) free(plane_recv_buffers[r]);
    }
    free(plane_send_buffers);
    free(plane_recv_buffers);
    free(send_buffer_offsets);
    free(send_counts_per_rank);
    free(recv_counts_per_rank);
    free(count_requests);
    free(data_requests);
}

void communicate_planes_alltoall(PlaneList* list, int N, int M, double x_max, double y_max, int rank, int size, int* tile_displacements) {
    // STEP 1: Initialize send buckets and counts per rank
    int* send_counts_per_rank = calloc(size, sizeof(int));
    MinPlaneToSend** plane_send_buckets = calloc(size, sizeof(MinPlaneToSend*));
    int* bucket_sizes_per_rank = calloc(size, sizeof(int));

    // STEP 2: Traverse the plane list and bucket planes for sending
    PlaneNode* current = list->head;
    while (current != NULL) {
        PlaneNode* next_node = current->next;
        int idx_i = get_index_i(current->x, x_max, N);
        int idx_j = get_index_j(current->y, y_max, M);
        int map_index = get_index(idx_i, idx_j, N, M);
        int dest_rank = get_rank_from_index(map_index, tile_displacements, size);

        if (dest_rank != rank) {
            // Pack plane data to send bucket
            MinPlaneToSend plane_info = {
                .index_plane = current->index_plane,
                .x = current->x,
                .y = current->y,
                .vx = current->vx,
                .vy = current->vy
            };
            bucket_sizes_per_rank[dest_rank]++;
            plane_send_buckets[dest_rank] = realloc(
                plane_send_buckets[dest_rank],
                bucket_sizes_per_rank[dest_rank] * sizeof(MinPlaneToSend)
            );
            plane_send_buckets[dest_rank][bucket_sizes_per_rank[dest_rank] - 1] = plane_info;
            remove_plane(list, current);
        }
        current = next_node;
    }

    // STEP 3:Prepare send counts array for MPI_Alltoall
    for (int r = 0; r < size; r++) {
        send_counts_per_rank[r] = bucket_sizes_per_rank[r];
    }

    // STEP 4: Exchange send/receive counts
    int* recv_counts_per_rank = calloc(size, sizeof(int));
    MPI_Alltoall(
        send_counts_per_rank, 1, MPI_INT,
        recv_counts_per_rank, 1, MPI_INT,
        MPI_COMM_WORLD
    );

    // STEP 5: Compute displacements and total counts
    int total_send = 0, total_recv = 0;
    int* send_displs = calloc(size, sizeof(int));
    int* recv_displs = calloc(size, sizeof(int));
    for (int r = 0; r < size; r++) {
        send_displs[r] = total_send;
        total_send += send_counts_per_rank[r];
        recv_displs[r] = total_recv;
        total_recv += recv_counts_per_rank[r];
    }

    // STEP 6: Allocate buffers for all-to-all data exchange
    MinPlaneToSend* send_buffer = malloc(total_send * sizeof(MinPlaneToSend));
    MinPlaneToSend* recv_buffer = malloc(total_recv * sizeof(MinPlaneToSend));

    // STEP 7: Flatten buckets into contiguous send buffer
    for (int r = 0, offset = 0; r < size; r++) {
        if (send_counts_per_rank[r] > 0) {
            memcpy(
                &send_buffer[offset],
                plane_send_buckets[r],
                send_counts_per_rank[r] * sizeof(MinPlaneToSend)
            );
            offset += send_counts_per_rank[r];
        }
        free(plane_send_buckets[r]);  // free each bucket
    }

    // STEP 8: Define MPI datatype for MinPlaneToSend
    MPI_Datatype MPI_MinPlane;
    MPI_Type_contiguous(sizeof(MinPlaneToSend) / sizeof(double),MPI_DOUBLE,&MPI_MinPlane);
    MPI_Type_commit(&MPI_MinPlane);

    // STEP 9: Perform all-to-all variable-length exchange
    MPI_Alltoallv(send_buffer, send_counts_per_rank, send_displs, MPI_MinPlane, recv_buffer, recv_counts_per_rank, recv_displs, MPI_MinPlane, MPI_COMM_WORLD);

    // STEP 10: Reinsert received planes into local list
    for (int i = 0; i < total_recv; i++) {
        MinPlaneToSend* p = &recv_buffer[i];
        int ii = get_index_i(p->x, x_max, N);
        int jj = get_index_j(p->y, y_max, M);
        int idx = get_index(ii, jj, N, M);
        insert_plane(
            list,
            p->index_plane,
            idx,
            rank,
            p->x,
            p->y,
            p->vx,
            p->vy
        );
    }

    // STEP 11: Cleanup allocations and MPI datatype
    free(send_counts_per_rank);
    free(bucket_sizes_per_rank);
    free(recv_counts_per_rank);
    free(send_displs);
    free(recv_displs);
    free(plane_send_buckets);
    free(send_buffer);
    free(recv_buffer);
    MPI_Type_free(&MPI_MinPlane);
}

void communicate_planes_struct_mpi(PlaneList* list,
                                   int N, int M,
                                   double x_max, double y_max,
                                   int rank, int size,
                                   int* tile_displacements)
{
    // STEP 1: Create and commit MPI datatype for MinPlaneToSend struct
    MPI_Datatype MPI_MinPlane;
    int blocklengths[5] = {1, 1, 1, 1, 1};
    MPI_Aint base_address;
    MPI_Aint displacements[5];
    
    MinPlaneToSend dummy_plane;

    MPI_Get_address(&dummy_plane, &base_address);
    MPI_Get_address(&dummy_plane.index_plane, &displacements[0]);
    MPI_Get_address(&dummy_plane.x,&displacements[1]);
    MPI_Get_address(&dummy_plane.y,&displacements[2]);
    MPI_Get_address(&dummy_plane.vx, &displacements[3]);
    MPI_Get_address(&dummy_plane.vy, &displacements[4]);

    for (int idx = 0; idx < 5; idx++) {
        displacements[idx] -= base_address;
    }

    MPI_Datatype types[5] = {MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    MPI_Type_create_struct(5, blocklengths, displacements, types, &MPI_MinPlane);
    MPI_Type_commit(&MPI_MinPlane);

    // STEP 2: Bucket planes to send per destination rank
    int* send_counts_per_rank = calloc(size, sizeof(int));
    MinPlaneToSend** plane_send_buckets = calloc(size, sizeof(MinPlaneToSend*));
    int* bucket_sizes_per_rank = calloc(size, sizeof(int));
    PlaneNode* current_node = list->head;

    while (current_node != NULL) {
        PlaneNode* next_node = current_node->next;
        int ii = get_index_i(current_node->x, x_max, N);
        int jj = get_index_j(current_node->y, y_max, M);
        int map_index = get_index(ii, jj, N, M);
        int dest_rank = get_rank_from_index(map_index, tile_displacements, size);

        if (dest_rank != rank) {
            //appending plane data to corresponding bucket
            bucket_sizes_per_rank[dest_rank]++;
            plane_send_buckets[dest_rank] = realloc(plane_send_buckets[dest_rank],bucket_sizes_per_rank[dest_rank] * sizeof(MinPlaneToSend));
            plane_send_buckets[dest_rank][bucket_sizes_per_rank[dest_rank] - 1] = (MinPlaneToSend){
                    .x = current_node->x,
                    .y = current_node->y,
                    .vx = current_node->vx,
                    .vy = current_node->vy
                    .index_plane = current_node->index_plane,
                };
            send_counts_per_rank[dest_rank]++;
            remove_plane(list, current_node);
        }
        current_node = next_node;
    }

    // STEP 3: Prepare send_counts_per_rank array for MPI_Alltoall
    for (int r = 0; r < size; r++) {
        send_counts_per_rank[r] = bucket_sizes_per_rank[r];
    }

    // STEP 4: Exchange number of planes to send/receive with all ranks
    int* recv_counts_per_rank = calloc(size, sizeof(int));
    MPI_Alltoall(
        send_counts_per_rank, 1, MPI_INT,
        recv_counts_per_rank, 1, MPI_INT,
        MPI_COMM_WORLD
    );

    // STEP 5: Compute displacements and total counts, allocate flat buffers
    int* send_displs = calloc(size, sizeof(int));
    int* recv_displs = calloc(size, sizeof(int));
    int total_send = 0, total_recv = 0;
    for (int r = 0; r < size; r++) {
        send_displs[r] = total_send;
        total_send += send_counts_per_rank[r];
        recv_displs[r] = total_recv;
        total_recv += recv_counts_per_rank[r];
    }
    MinPlaneToSend* recv_buffer = malloc(total_recv * sizeof(MinPlaneToSend));
    MinPlaneToSend* send_buffer = malloc(total_send * sizeof(MinPlaneToSend));
    

    for (int r = 0, offset = 0; r < size; r++) {
        if (send_counts_per_rank[r] > 0) {
            memcpy(
                &send_buffer[offset],
                plane_send_buckets[r],
                send_counts_per_rank[r] * sizeof(MinPlaneToSend)
            );
            offset += send_counts_per_rank[r];
        }
        free(plane_send_buckets[r]);
    }

    // STEP 6: Perform all-to-all variable-length communication
    MPI_Alltoallv(send_buffer, send_counts_per_rank, send_displs, MPI_MinPlane,recv_buffer, recv_counts_per_rank, recv_displs, MPI_MinPlane,MPI_COMM_WORLD);

    // STEP 7: Insert all received planes back into the local list
    for (int i = 0; i < total_recv; i++) {
        MinPlaneToSend* p = &recv_buffer[i];
        int xi = get_index_i(p->x, x_max, N);
        int yi = get_index_j(p->y, y_max, M);
        int idx = get_index(xi, yi, N, M);
        insert_plane(list,p->index_plane,idx,rank,p->x, p->y, p->vx, p->vy);
    }

    // STEP 8: Cleanup all allocated buffers and free MPI type
    free(send_counts_per_rank);
    free(bucket_sizes_per_rank);
    free(recv_counts_per_rank);
    free(send_displs);
    free(recv_displs);
    free(plane_send_buckets);
    free(send_buffer);
    free(recv_buffer);
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


