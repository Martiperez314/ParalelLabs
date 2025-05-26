#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#define BLOCKSIZE 128

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return 1; \
    } \
} while (0)

// Kernel definition
__global__ void vecadd_cuda(double *A, double *B, double *C, const int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) {
        C[index] = A[index] + B[index];
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: ./vecadd_cuda <vector size N>\n");
        return 1;
    }

    int N = atoi(argv[1]);
    size_t size = N * sizeof(double);

    printf("Vector size: %d\n", N);

    // Host memory allocation and initialization
    double *A = (double *)malloc(size);
    double *B = (double *)malloc(size);
    double *C = (double *)malloc(size);
    for (int i = 0; i < N; i++) {
        A[i] = (double)i;
        B[i] = 2.0 * (N - i);
    }

    // Device memory allocation
    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    // Timing variables
    cudaEvent_t start, stop;
    float time_h2d, time_kernel, time_d2h;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Host to Device Copy Timing
    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_h2d, start, stop);

    // Kernel Timing
    int blocksPerGrid = (N + BLOCKSIZE - 1) / BLOCKSIZE;
    cudaEventRecord(start);
    vecadd_cuda<<<blocksPerGrid, BLOCKSIZE>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_kernel, start, stop);

    // Device to Host Copy Timing
    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_d2h, start, stop);

    // Validation
    const double tol = 1e-6;
    for (int i = 0; i < N; ++i) {
        double expected = 2.0 * N - i;
        if (fabs(C[i] - expected) > tol) {
            fprintf(stderr, "Validation failed at index %d: C[%d] = %f, expected %f\n",
                    i, i, C[i], expected);
            free(A);
            free(B);
            free(C);
            return EXIT_FAILURE;
        }
    }

    // Final Output Format as specified
    printf("Copy A and B Host to Device elapsed time: %.9f seconds\n", time_h2d / 1000.0);
    printf("Kernel elapsed time: %.9f seconds\n", time_kernel / 1000.0);
    printf("Copy C Device to Host elapsed time: %.9f seconds\n", time_d2h / 1000.0);
    printf("Total elapsed time: %.9f seconds\n", (time_h2d + time_kernel + time_d2h) / 1000.0);

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(A);
    free(B);
    free(C);

    return 0;
}
