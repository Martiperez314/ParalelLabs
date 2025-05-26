#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// OpenACC vector addition
void vecadd_oacc(double *A, double *B, double *C, const int N) {
    #pragma acc parallel loop present(A[0:N], B[0:N], C[0:N])
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <vector size N>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    printf("Vector size: %d\n", N);

    double *A = (double *)malloc(N * sizeof(double));
    double *B = (double *)malloc(N * sizeof(double));
    double *C = (double *)malloc(N * sizeof(double));
    if (!A || !B || !C) {
        fprintf(stderr, "Error: memory allocation failed.\n");
        free(A); free(B); free(C);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < N; ++i) {
        A[i] = (double)i;
        B[i] = 2.0 * (N - i);
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Manual memory transfer control
    #pragma acc data copyin(A[0:N], B[0:N]) copyout(C[0:N])
    {
        vecadd_oacc(A, B, C, N);
        #pragma acc wait
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1.0e9;
    printf("Elapsed time: %.9f seconds\n", elapsed);

    const double tol = 1e-6;
    for (int i = 0; i < N; ++i) {
        double expected = 2.0 * N - i;
        if (fabs(C[i] - expected) > tol) {
            fprintf(stderr, "Validation failed at index %d: C[%d] = %f, expected %f\n",
                    i, i, C[i], expected);
            free(A); free(B); free(C);
            return EXIT_FAILURE;
        }
    }

    free(A); free(B); free(C);
    return 0;
}
