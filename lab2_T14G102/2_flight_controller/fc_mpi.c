#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include "auxiliar.h"

// Reading the planes from a file for MPI
// Each rank reads the full file, but only stores the planes it is responsible for

void read_planes_mpi(const char* filename, PlaneList* planes, int* N, int* M, double* x_max, double* y_max, int rank, int size, int* tile_displacements)
{
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return;
    }

    char line[MAX_LINE_LENGTH];
    int num_planes = 0;

    // Reading header
    fgets(line, sizeof(line), file);
    fgets(line, sizeof(line), file);
    sscanf(line, "# Map: %lf, %lf : %d %d", x_max, y_max, N, M);
    fgets(line, sizeof(line), file);
    sscanf(line, "# Number of Planes: %d", &num_planes);
    fgets(line, sizeof(line), file);
    
    //Flight Controller: Domain decomposition (MIP)
    for(int i = 0; i < size + 1; i++){
        tile_displacements[i] = i * (N * M) / size;
    }

    // Reading plane data
    int index = 0;
    while (fgets(line, sizeof(line), file)) {
        int idx;
        double x, y, vx, vy;
        if (sscanf(line, "%d %lf %lf %lf %lf", &idx, &x, &y, &vx, &vy) == 5) {
            int index_i = get_index_i(x, *x_max, *N);
            int index_j = get_index_j(y, *y_max, *M);
            int index_map = get_index(index_i, index_j, *N, *M);
            if (index_map >= tile_displacements[rank] && index_map < tile_displacements[rank + 1]) {
                insert_plane(planes, idx, index_map, rank, x, y, vx, vy);
                index++;
            }
        }
    }
    fclose(file);

    printf("Total planes read: %d\n", index);
    assert(num_planes == index);
}

/// TODO
/// Communicate planes using mainly Send/Recv calls with default data types
void communicate_planes_send(PlaneList* list, int N, int M, double x_max, double y_max, int rank, int size, int* tile_displacements)
{
    int* n_planes_to_send = (int*)calloc(size, sizeof(int));// Allocate an array to count how many planes each process will receive
    int total_planes_to_send = 0;
    PlaneList* planes_to_send = (NULL, NULL);// Create a new list to hold planes that need to be sent to other processes
    PlaneNode* current = list->head;// Pointer to the start of the local plane list

    while (current != NULL) {
        // Determine which process (rank) should now handle this plane based on its position
        int new_rank = get_rank_from_indices(current->x, current->y, N, M, tile_displacements, size);
        if (new_rank != rank) { // If the plane does not belong to the current process anymore then increment the count of planes to be sent to this new_rank
            n_planes_to_send[new_rank]++;

            // Then you add this plane and remove it from local list 
            insert_plane(planes_to_send, current->index_plane, current->index_map, rank, current->x, current->y, current->vx, current->vy);
            remove_plane(list, current);
            total_planes_to_send++;// Keep track of how many total planes need to be sent
        }
        current = current->next;// Move to the next plane in the list
    }

    // We prepare non-blocking send requests for sending plane counts and send the number of planes that will be sent to each process
    MPI_Request* req = (MPI_Request*)malloc(sizeof(MPI_Request) * size);
    for (int i = 0; i < size; i++) {
        MPI_Isend(&n_planes_to_send[i], 1, MPI_INT, i, MPI_PLANE, MPI_COMM_WORLD, &req[i]);
    }

    int planes_to_receive = 0;
    int aux;
    MPI_Status status;

    // Receive how many planes this process will receive from any source
    for (int i = 0; i < size; i++) {
        MPI_Recv(&aux, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        planes_to_receive += aux;  // Accumulate the total number of planes to expect
    }
    free(n_planes_to_send);
    free(req);

    // Allocate new requests for sending actual plane data
    req = (MPI_Request*)malloc(sizeof(MPI_Request) * total_planes_to_send);
    current = planes_to_send->head; // Reset pointer to start of planes_to_send list
    int aux2 = 0;

    // Allocate memory to store all planes being sent (5 values per plane)
    double* planes = (double*)malloc(sizeof(double) * total_planes_to_send * 5);

    // For each plane in the send list, prepare and send its data
    while (current != NULL) {
        // Create an array to hold this plane's data
        double* plane = (double*)malloc(sizeof(double) * 5); 
        plane[0] = (double)(current->index_plane);  // Transform int to double (later it will be treansform it back)
        plane[1] = current->x;                      
        plane[2] = current->y;                     
        plane[3] = current->vx;                     
        plane[4] = current->vy;                     

        // Get the process rank and send it to the actual plane data to the correct destination
        int new_rank = get_rank_from_indices(current->x, current->y, N, M, tile_displacements, size);
        MPI_Isend(plane, 5, MPI_DOUBLE, new_rank, MPI_PLANE, MPI_COMM_WORLD, &req[aux2]);
        aux2++;
        current = current->next;
    }

    double plane[5];  // Then we recive actual planes
    for (int i = 0; i < planes_to_receive; i++) {
        MPI_Recv(&plane[0], 5, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        insert_plane(list, (int)plane[0], rank, plane[1], plane[2], plane[3], plane[4]);// Insert the received plane into the local list
    }
    free(req);
}
/// TODO
/// Communicate planes using all to all calls with default data types
void communicate_planes_alltoall(PlaneList* list, int N, int M, double x_max, double y_max, int rank, int size, int* tile_displacements)
{
    // 1) Build temporary list of planes to send and count per‐dest
    int *send_counts = calloc(size, sizeof(int));
    PlaneList to_send = { NULL, NULL };

    // We need to safely remove nodes while iterating
    PlaneNode *cur = list->head, *next;
    while (cur) {
        next = cur->next;
        int dest = get_rank_from_indices(cur->x, cur->y,N, M,tile_displacements,size);
        if (dest != rank) {
            send_counts[dest]++;
            // move node into to_send list
            insert_plane(&to_send,cur->index_plane,cur->index_map,rank,cur->x, cur->y,cur->vx, cur->vy);
            remove_plane(list, cur);
        }
        cur = next;
    }

    // 2) Exchange counts to get recv_counts
    int *recv_counts = calloc(size, sizeof(int));
    MPI_Alltoall(send_counts, 1, MPI_INT,recv_counts, 1, MPI_INT,MPI_COMM_WORLD);

    // 3) Build displacements (in doubles) and total lengths
    int *send_disp = malloc(size * sizeof(int));
    int *recv_disp = malloc(size * sizeof(int));
    send_disp[0] = recv_disp[0] = 0;
    for (int i = 1; i < size; i++) {
        send_disp[i] = send_disp[i-1] + send_counts[i-1] * 5;
        recv_disp[i] = recv_disp[i-1] + recv_counts[i-1] * 5;
    }
    int total_send = send_disp[size-1] + send_counts[size-1] * 5;
    int total_recv = recv_disp[size-1] + recv_counts[size-1] * 5;

    // 4) Pack send buffer
    double *send_buf = malloc(total_send * sizeof(double));
    int *temp_idx = calloc(size, sizeof(int));
    cur = to_send.head;
    while (cur) {
        int dest = get_rank_from_indices(cur->x, cur->y,N, M,tile_displacements,size);
        int offset = send_disp[dest] + temp_idx[dest] * 5;
        send_buf[offset + 0] = (double)cur->index_plane;
        send_buf[offset + 1] = cur->x;
        send_buf[offset + 2] = cur->y;
        send_buf[offset + 3] = cur->vx;
        send_buf[offset + 4] = cur->vy;
        temp_idx[dest]++;
        cur = cur->next;
    }
    free(temp_idx);

    // 5) Alltoallv exchange
    double *recv_buf = malloc(total_recv * sizeof(double));
    // Note: send_counts and recv_counts are in #planes, but Alltoallv needs counts in #doubles:
    int *send_counts_d = malloc(size*sizeof(int));
    int *recv_counts_d = malloc(size*sizeof(int));
    for (int i = 0; i < size; i++) {
        send_counts_d[i] = send_counts[i] * 5;
        recv_counts_d[i] = recv_counts[i] * 5;
    }
    MPI_Alltoallv(send_buf, send_counts_d, send_disp, MPI_DOUBLE,recv_buf, recv_counts_d, recv_disp, MPI_DOUBLE,MPI_COMM_WORLD);

    // 6) Unpack received planes into main list
    for (int i = 0; i < total_recv / 5; i++) {
        int idx = (int)recv_buf[5*i + 0];
        double x_val = recv_buf[5*i + 1];
        double y_val =recv_buf[5*i + 2];
        double vx_val=recv_buf[5*i + 3];
        double vy_val =recv_buf[5*i + 4];
        // recompute map index
        int ii = get_index_i(x_val, x_max, N);
        int jj = get_index_j(y_val, y_max, M);
        int idx_map = get_index(ii, jj, N, M);
        insert_plane(list, idx, idx_map, rank, x_val, y_val, vx_val, vy_val);
    }

    // 7) Cleanup
    free(send_counts);
    free(recv_counts);
    free(send_disp);
    free(recv_disp);
    free(send_buf);
    free(recv_buf);
    free(send_counts_d);
    free(recv_counts_d);
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
    MinPlaneToSend dummy;
    MPI_Datatype MPI_MinPlaneType;
    {
        int blocklengths[5] = {1, 1, 1, 1, 1};
        MPI_Aint displacements[5];
        MPI_Datatype types[5] = { MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
        MPI_Aint base;
        MPI_Get_address(&dummy, &base);
        MPI_Get_address(&dummy.index_plane, &displacements[0]);
        MPI_Get_address(&dummy.x,           &displacements[1]);
        MPI_Get_address(&dummy.y,           &displacements[2]);
        MPI_Get_address(&dummy.vx,          &displacements[3]);
        MPI_Get_address(&dummy.vy,          &displacements[4]);
        for (int i = 0; i < 5; i++) {
            displacements[i] -= base;
        }
        MPI_Type_create_struct(5, blocklengths, displacements, types, &MPI_MinPlaneType);
        MPI_Type_commit(&MPI_MinPlaneType);
    }

    // 1) Collect planes that moved off this rank into to_send list
    int *send_counts = calloc(size, sizeof(int));
    PlaneList to_send = { NULL, NULL };
    PlaneNode *cur = list->head;
    PlaneNode *next;
    while (cur) {
        next = cur->next;
        int dest = get_rank_from_indices(cur->x, cur->y, N, M, tile_displacements, size);
        if (dest != rank) {
            send_counts[dest]++;
            insert_plane(&to_send,
                         cur->index_plane,
                         cur->index_map,
                         rank,
                         cur->x, cur->y,
                         cur->vx, cur->vy);
            remove_plane(list, cur);
        }
        cur = next;
    }

    // 2) Exchange counts
    int *recv_counts = calloc(size, sizeof(int));
    MPI_Alltoall(send_counts, 1, MPI_INT,
                 recv_counts, 1, MPI_INT,
                 MPI_COMM_WORLD);

    // 3) Build displacements in number of structs
    int *send_disp = malloc(size * sizeof(int));
    int *recv_disp = malloc(size * sizeof(int));
    send_disp[0] = recv_disp[0] = 0;
    for (int i = 1; i < size; i++) {
        send_disp[i] = send_disp[i-1] + send_counts[i-1];
        recv_disp[i] = recv_disp[i-1] + recv_counts[i-1];
    }
    int total_send = send_disp[size-1] + send_counts[size-1];
    int total_recv = recv_disp[size-1] + recv_counts[size-1];

    // 4) Pack into buffer of structs
    MinPlaneToSend *send_buf = malloc(total_send * sizeof(MinPlaneToSend));
    int *temp_idx = calloc(size, sizeof(int));
    cur = to_send.head;
    while (cur) {
        int dest = get_rank_from_indices(cur->x, cur->y, N, M, tile_displacements, size);
        int pos = send_disp[dest] + temp_idx[dest];
        send_buf[pos].index_plane = cur->index_plane;
        send_buf[pos].x           = cur->x;
        send_buf[pos].y           = cur->y;
        send_buf[pos].vx          = cur->vx;
        send_buf[pos].vy          = cur->vy;
        temp_idx[dest]++;
        cur = cur->next;
    }
    free(temp_idx);

    // 5) Alltoallv with struct datatype
    MinPlaneToSend *recv_buf = malloc(total_recv * sizeof(MinPlaneToSend));
    MPI_Alltoallv(send_buf, send_counts, send_disp, MPI_MinPlaneType,
                  recv_buf, recv_counts, recv_disp, MPI_MinPlaneType,
                  MPI_COMM_WORLD);

    // 6) Unpack received planes into main list
    for (int i = 0; i < total_recv; i++) {
        MinPlaneToSend *mp = &recv_buf[i];
        int    idx   = mp->index_plane;
        double x_val = mp->x;
        double y_val = mp->y;
        double vx_val= mp->vx;
        double vy_val= mp->vy;
        int ii = get_index_i(x_val, x_max, N);
        int jj = get_index_j(y_val, y_max, M);
        int idx_map = get_index(ii, jj, N, M);
        insert_plane(list, idx, idx_map, rank,
                     x_val, y_val, vx_val, vy_val);
    }

    // 7) Cleanup
    free(send_counts);
    free(recv_counts);
    free(send_disp);
    free(recv_disp);
    free(send_buf);
    free(recv_buf);
    MPI_Type_free(&MPI_MinPlaneType);
    // lo mismo q alltoall pero cambiando una cosa, tenemos q crear propio datatype (lo puedes crear) i configurarlo (tutorial) web: https://rookiehpc.org/mpi/docs/mpi_type_create_struct/index.html
}

int main(int argc, char **argv) {
    int debug = 0;                      // 0: no debug, 1: shows all planes information during checking
    int N = 0, M = 0;                   // Grid dimensions
    double x_max = 0.0, y_max = 0.0;    // Total grid size
    int max_steps;                      // Total simulation steps
    char* input_file;                   // Input file name
    int check;                          // 0: no check, 1: check the simulation is correct

    int rank, size;

    // Initialize MPI environment and get rank and size
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Allocate tile displacement vector with size + 1 entries
    int* tile_displacements = (int*)malloc((size + 1) * sizeof(int));
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

    // Debug: Print planes per process to verify correct reading
    print_planes_par_debug(&owning_planes);

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

    if (rank == 0) {
        printf("Flight controller simulation: #input %s mode: %d size: %d\n", input_file, mode, size);
        printf("Time simulation:     %.2fs\n", time_sim);
        printf("Time communication:  %.2fs\n", time_comm);
        printf("Time total:          %.2fs\n", time_total);
    }

    if (check == 1)
        check_planes_mpi(&owning_planes_t0, &owning_planes, N, M, x_max, y_max, max_steps, tile_displacements, size, debug);

    MPI_Finalize();
    free(tile_displacements);
    return 0;
}
