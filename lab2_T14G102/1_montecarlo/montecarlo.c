#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <math.h> 

#define M_PI 3.14159265358979323846 

// Random number generator structure and function
typedef struct { uint64_t state; uint64_t inc; } pcg32_random_t;
double pcg32_random(pcg32_random_t* rng) {
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc|1);
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    uint32_t ran_int = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    return (double)ran_int / (double)UINT32_MAX;
}

int main(int argc, char **argv) {
    int N;
    long NUM_SAMPLES, SEED;

    if (argc == 4) {
        N = atoi(argv[1]);
        NUM_SAMPLES = atol(argv[2]);
        SEED = atol(argv[3]);
    } else {
        N = 3;
        NUM_SAMPLES = 1000000;
        SEED = time(NULL);
    }

    MPI_Init(&argc, &argv); // Initialize MPI

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Who am I?
    MPI_Comm_size(MPI_COMM_WORLD, &size); // How many are we?

    // Initialize random number generator
    pcg32_random_t rng;
    rng.state = SEED + rank;
    rng.inc = (rank << 16) | 0x3039;

    // Divide samples evenly among processes
    long local_samples = NUM_SAMPLES / size;
    if (rank < NUM_SAMPLES % size) {
        local_samples += 1; // Distribute remaining samples
    }

    // Start timing
    double start_time = MPI_Wtime();

    long local_inside = 0; // Local count of points inside hypersphere

    // Monte Carlo sampling loop
    for (long i = 0; i < local_samples; i++) {
        double sum = 0.0;
        for (int d = 0; d < N; d++) {
            double x = 2.0 * pcg32_random(&rng) - 1.0; // Random number in [-1, 1]
            sum += x * x;
        }
        if (sum <= 1.0) {
            local_inside++; // Point is inside the hypersphere
        }
    }

    // End timing
    double end_time = MPI_Wtime();
    double local_elapsed = end_time - start_time;

    // Reduce results: sum points inside across all processes
    long total_inside;
    MPI_Reduce(&local_inside, &total_inside, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // Reduce maximum elapsed time across all processes
    double global_elapsed;
    MPI_Reduce(&local_elapsed, &global_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Output results on the root process
    if (rank == 0) {
        double phi_estimate = (double)total_inside / (double)NUM_SAMPLES;
        double theoretical = pow(M_PI, N/2.0) / tgamma(N/2.0 + 1.0) / pow(2.0, N);
        double rel_error = fabs(phi_estimate - theoretical) / theoretical;
        
        printf("Monte Carlo sphere/cube ratio estimation\n");
        printf("N: %ld samples, d: %d, seed: %ld, size: %d\n", NUM_SAMPLES, N, SEED, size);
        printf("Ratio =  %.3e (%.3e) Err: %.3e\n", phi_estimate, theoretical, rel_error);
        printf("Elapsed time: %.3f seconds\n", global_elapsed);
    }

    MPI_Finalize(); // Finalize MPI
    return 0;
}