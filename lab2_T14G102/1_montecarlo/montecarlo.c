#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <math.h> 

#define M_PI 3.14159265358979323846 

// Here the struct for PCG32-based random number generator
typedef struct {
    uint64_t state;
    uint64_t inc;
} pcg32_random_t;

// We generates a double in [0, 1] using PCG32 RNG
double get_random_number(pcg32_random_t* rng) {
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc | 1);
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    uint32_t random_bits = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    return (double)random_bits / (double)UINT32_MAX;
}

int main(int argc, char **argv) {
    // We initialize default values
    int dimension = 3;
    long total_points = 1000000;
    long seed = time(NULL);

    // Then if arguments are provided, override defaults
    if (argc == 4) {
        dimension = atoi(argv[1]);
        total_points = atol(argv[2]);
        seed = atol(argv[3]);
    }
    
    MPI_Init(&argc, &argv);// Initialize the MPI environment
    // We determine the rank of this process and total number of processes
    int my_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    pcg32_random_t rng; //Then, initialize RNG with different seed per process to ensure randomness
    rng.state = seed + my_rank;
    rng.inc = (my_rank << 16) | 0x3039;

    // Divide the workload (samples) equally among processes
    long points_per_process = total_points / world_size;
    if (my_rank < total_points % world_size) {
        points_per_process += 1; // Here we have to give extra sample to first few ranks
    }
    // Start measuring execution time for this process
    double start_time = MPI_Wtime();
    long points_inside_sphere = 0;

    // Then we perform Monte Carlo estimation
    for (long sample = 0; sample < points_per_process; sample++) {
        double distance_squared = 0.0;

        // Generate a random point in N-dimensional space
        for (int d = 0; d < dimension; d++) {
            double coord = 2.0 * get_random_number(&rng) - 1.0; // Scale to [-1, 1]
            distance_squared += coord * coord; // Compute squared Euclidean distance
        }
        // Count the point if it lies inside the unit hypersphere
        if (distance_squared <= 1.0) {
            points_inside_sphere++;
        }
    }
    // Measure end time for current process
    double end_time = MPI_Wtime();
    double time_taken = end_time - start_time;

    // We combine all counts into a global count at rank 0
    long global_inside;
    MPI_Reduce(&points_inside_sphere, &global_inside, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    // Fnd the maximum execution time among all processes
    double max_time;
    MPI_Reduce(&time_taken, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Only the root process prints the final results
    if (my_rank == 0) {
        double volume_ratio_estimate = (double)global_inside / (double)total_points;

        // Exact theoretical value of the ratio between hypersphere and hypercube
        double true_ratio = pow(M_PI, dimension / 2.0) /
                            tgamma(dimension / 2.0 + 1.0) /
                            pow(2.0, dimension);

        double relative_error = fabs(volume_ratio_estimate - true_ratio) / true_ratio;

        printf("Monte Carlo sphere/cube ratio estimation\n");
        printf("Samples: %ld, Dimensions: %d, Seed: %ld, Processes: %d\n",
               total_points, dimension, seed, world_size);
        printf("Estimated Ratio: %.3e (True: %.3e), Relative Error: %.3e\n",
               volume_ratio_estimate, true_ratio, relative_error);
        printf("Max Elapsed Time: %.3f seconds\n", max_time);
    }

    MPI_Finalize();
    return 0;
}
