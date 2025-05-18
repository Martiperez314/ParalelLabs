#include <stdio.h>

#include <assert.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#include "auxiliar.h"

/// TODO
/// Reading the planes from a file for MPI
void read_planes_mpi(const char* filename, PlaneList* planes, int* N, int* M, double* x_max, double* y_max, int rank, int size, int* tile_displacements)
{
    PlaneList all_planes = {NULL, NULL};

    if (rank == 0) {
        read_planes_seq_full(filename, &all_planes, N, M, x_max, y_max);

        PlaneNode* current = all_planes.head;
        while (current != NULL) {
            int owner_rank = get_rank_from_index(current->index_map, tile_displacements, size);

            if (owner_rank == 0) {
                insert_plane(planes, current->index_plane, current->index_map, current->rank, current->x, current->y, current->vx, current->vy);
            } else {
                double data[6] = {
                    (double)current->index_plane,
                    current->x, current->y,
                    current->vx, current->vy,
                    (double)current->index_map
                };
                MPI_Send(data, 6, MPI_DOUBLE, owner_rank, 0, MPI_COMM_WORLD);
            }

            current = current->next;
        }

        // Send termination signal to other ranks
        double end_signal[1] = {-1};
        for (int r = 1; r < size; ++r) {
            MPI_Send(end_signal, 1, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
        }
    } else {
        while (1) {
            double data[6];
            MPI_Recv(data, 6, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (data[0] == -1.0) break;

            int idx = (int)data[0];
            double x = data[1], y = data[2];
            double vx = data[3], vy = data[4];
            int index_map = (int)data[5];

            insert_plane(planes, idx, index_map, rank, x, y, vx, vy);
        }
    }
}

/// TODO
/// Communicate planes using mainly Send/Recv calls with default data types
void communicate_planes_send(PlaneList* list, int N, int M, double x_max, double y_max, int rank, int size, int* tile_displacements)
{
    PlaneNode* current = list->head;
    while (current != NULL) {
        PlaneNode* next = current->next;
        current->x += current->vx;
        current->y += current->vy;

        if (current->x <= 1e-3 || current->x >= (x_max - 1e-3) ||
            current->y <= 1e-3 || current->y >= (y_max - 1e-3)) {
            remove_plane(list, current);
        } else {
            int new_i = get_index_i(current->x, x_max, N);
            int new_j = get_index_j(current->y, y_max, M);
            int new_index_map = get_index(new_i, new_j, N, M);
            int new_rank = get_rank_from_index(new_index_map, tile_displacements, size);

            if (new_rank == rank) {
                current->index_map = new_index_map;
            } else {
                double data[6] = {
                    (double)current->index_plane,
                    current->x, current->y,
                    current->vx, current->vy,
                    (double)new_index_map
                };
                MPI_Send(data, 6, MPI_DOUBLE, new_rank, 0, MPI_COMM_WORLD);
                remove_plane(list, current);
            }
        }

        current = next;
    }

    // Receive incoming planes
    MPI_Status status;
    int flag;
    do {
        MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
        if (flag) {
            double data[6];
            MPI_Recv(data, 6, MPI_DOUBLE, status.MPI_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int idx = (int)data[0];
            insert_plane(list, idx, (int)data[5], rank, data[1], data[2], data[3], data[4]);
        }
    } while (flag);
}

/// TODO
/// Communicate planes using all to all calls with default data types
void communicate_planes_alltoall(PlaneList* list, int N, int M, double x_max, double y_max, int rank, int size, int* tile_displacements)
{
    MinPlaneToSend* sendbuf[size];
    int sendcounts[size], sdispls[size];
    memset(sendcounts, 0, sizeof(sendcounts));
    memset(sdispls, 0, sizeof(sdispls));

    int total_send = 0;

    PlaneNode* current = list->head;
    while (current != NULL) {
        PlaneNode* next = current->next;

        current->x += current->vx;
        current->y += current->vy;

        if (current->x <= 1e-3 || current->x >= (x_max - 1e-3) ||
            current->y <= 1e-3 || current->y >= (y_max - 1e-3)) {
            remove_plane(list, current);
        } else {
            int new_i = get_index_i(current->x, x_max, N);
            int new_j = get_index_j(current->y, y_max, M);
            int new_index_map = get_index(new_i, new_j, N, M);
            int new_rank = get_rank_from_index(new_index_map, tile_displacements, size);

            if (new_rank == rank) {
                current->index_map = new_index_map;
            } else {
                if (!sendcounts[new_rank]) {
                    sendbuf[new_rank] = malloc(sizeof(MinPlaneToSend) * 1000);
                }
                sendbuf[new_rank][sendcounts[new_rank]++] = (MinPlaneToSend){
                    current->index_plane, current->x, current->y, current->vx, current->vy
                };
                remove_plane(list, current);
            }
        }

        current = next;
    }

    int recvcounts[size];
    MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);

    int total_recv = 0;
    for (int i = 0; i < size; i++) {
        total_recv += recvcounts[i];
    }

    MinPlaneToSend* recvbuf = malloc(sizeof(MinPlaneToSend) * total_recv);
    MPI_Datatype plane_type;
    MPI_Type_contiguous(5, MPI_DOUBLE, &plane_type);
    MPI_Type_commit(&plane_type);

    MinPlaneToSend* flat_sendbuf = malloc(sizeof(MinPlaneToSend) * total_send);
    int offset = 0;
    for (int i = 0; i < size; i++) {
        sdispls[i] = offset;
        for (int j = 0; j < sendcounts[i]; j++) {
            flat_sendbuf[offset++] = sendbuf[i][j];
        }
        free(sendbuf[i]);
    }

    int rdispls[size];
    rdispls[0] = 0;
    for (int i = 1; i < size; i++)
        rdispls[i] = rdispls[i-1] + recvcounts[i-1];

    MPI_Alltoallv(flat_sendbuf, sendcounts, sdispls, plane_type,
                  recvbuf, recvcounts, rdispls, plane_type, MPI_COMM_WORLD);

    for (int i = 0; i < total_recv; i++) {
        MinPlaneToSend p = recvbuf[i];
        int i_map = get_index_i(p.x, x_max, N);
        int j_map = get_index_j(p.y, y_max, M);
        int index_map = get_index(i_map, j_map, N, M);
        insert_plane(list, p.index_plane, index_map, rank, p.x, p.y, p.vx, p.vy);
    }

    MPI_Type_free(&plane_type);
    free(flat_sendbuf);
    free(recvbuf);
}

typedef struct {
    int    index_plane;
    double x;
    double y;
    double vx;
    double vy;
} MinPlaneToSend;

/// TODO
/// Communicate planes using all to all calls with custom data types
void communicate_planes_struct_mpi(PlaneList* list,
                               int N, int M,
                               double x_max, double y_max,
                               int rank, int size,
                               int* tile_displacements)
{
    communicate_planes_alltoall(list, N, M, x_max, y_max, rank, size, tile_displacements);
}

int main(int argc, char **argv) {
    int debug = 0;                      // 0: no debug, 1: shows all planes information during checking
    int N = 0, M = 0;                   // Grid dimensions
    double x_max = 0.0, y_max = 0.0;    // Total grid size
    int max_steps;                      // Total simulation steps
    char* input_file;                   // Input file name
    int check;                          // 0: no check, 1: check the simulation is correct

    int rank, size;

    /// TODO
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int tile_displacements[size+1];
    int mode = 0;
    if (argc == 5) {
        input_file = argv[1];
        max_steps = atoi(argv[2]);
        if (max_steps <= 0) {
            fprintf(stderr, "max_steps needs to be a positive integer\n");
            return 1;
        }
        mode = atoi(argv[3]);
        if (mode > 2 || mode < 0) {
            fprintf(stderr, "mode needs to be a value between 0 and 2\n");
            return 1;
        }
        check = atoi(argv[4]);
        if (check >= 2 || check < 0) {
            fprintf(stderr, "check needs to be a 0 or 1\n");
            return 1;
        }
    }
    else {
        fprintf(stderr, "Usage: %s <filename> <max_steps> <mode> <check>\n", argv[0]);
        return 1;
    }

    PlaneList owning_planes = {NULL, NULL};
    read_planes_mpi(input_file, &owning_planes, &N, &M, &x_max, &y_max, rank, size, tile_displacements);
    PlaneList owning_planes_t0 = copy_plane_list(&owning_planes);

    //print_planes_par_debug(&owning_planes);

    double time_sim = 0., time_comm = 0, time_total=0;

    double start_time = MPI_Wtime();
    int step = 0;
    for (step = 1; step <= max_steps; step++) {
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

    /// TODO Check computational times


    if (rank == 0) {
        printf("Flight controller simulation: #input %s mode: %d size: %d\n", input_file, mode, size);
        printf("Time simulation:     %.2fs\n", time_sim);
        printf("Time communication:  %.2fs\n", time_comm);
        printf("Time total:          %.2fs\n", time_total);
    }

    if (check ==1)
        check_planes_mpi(&owning_planes_t0, &owning_planes, N, M, x_max, y_max, max_steps, tile_displacements, size, debug);

    MPI_Finalize();
    return 0;
}
