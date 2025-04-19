#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#define _USE_MATH_DEFINES 
#include <math.h>

void initialize(double *v, int N) {
    for (int i = 0; i < N; i++) {
        v[i] = (1 - pow(0.5 - (double)i / (double)N, 2)) * cos(2 * M_PI * 100 * (i - 0.5) / N);
    }
}

void argmax_seq(double *v, int N, double *m, int *idx_m) {
    *m = v[0];
    *idx_m = 0;
    for (int i = 1; i < N; i++) {
        if (v[i] > *m) {
            *m = v[i];
            *idx_m = i;
        }
    }
}

void argmax_par(double *v, int N, double *m, int *idx_m) {
    double local_max = v[0];
    int local_idx = 0;
    #pragma omp parallel
    {
        double thread_max = local_max;
        int thread_idx = local_idx;
        #pragma omp for nowait
        for (int i = 1; i < N; i++) {
            if (v[i] > thread_max) {
                thread_max = v[i];
                thread_idx = i;
            }
        }
        #pragma omp critical
        {
            if (thread_max > local_max) {
                local_max = thread_max;
                local_idx = thread_idx;
            }
        }
    }
    *m = local_max;
    *idx_m = local_idx;
}

void argmax_recursive(double *v, int N, double *m, int *idx_m, int K) {
    if (N < K) {
        argmax_seq(v, N, m, idx_m);
        return;
    }
    double m1, m2;
    int idx1, idx2;
    int mid = N / 2;
    argmax_recursive(v, mid, &m1, &idx1, K);
    argmax_recursive(v + mid, N - mid, &m2, &idx2, K);

    if (m1 >= m2) {
        *m = m1;
        *idx_m = idx1;
    } else {
        *m = m2;
        *idx_m = idx2 + mid;
    }
}

void argmax_recursive_tasks(double *v, int N, double *m, int *idx_m, int K) {
    if (N < K) {
        argmax_seq(v, N, m, idx_m);
        return;
    }
    double m1, m2;
    int idx1, idx2;
    int mid = N / 2;

    #pragma omp task shared(m1, idx1)
    argmax_recursive_tasks(v, mid, &m1, &idx1, K);

    #pragma omp task shared(m2, idx2)
    argmax_recursive_tasks(v + mid, N - mid, &m2, &idx2, K);

    #pragma omp taskwait

    if (m1 >= m2) {
        *m = m1;
        *idx_m = idx1;
    } else {
        *m = m2;
        *idx_m = idx2 + mid;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <nthreads> <K>\n", argv[0]);
        return 1;
    }
    int nthreads = atoi(argv[1]);
    int K = atoi(argv[2]);
    const int N = 40962;
    omp_set_num_threads(nthreads);

    double *v = malloc(N * sizeof(double));
    initialize(v, N);

    double seq_m, par_m, rec_m, task_m;
    int seq_idx_m, par_idx_m, rec_idx_m, task_idx_m;
    double start, end;

    start = omp_get_wtime();
    argmax_seq(v, N, &seq_m, &seq_idx_m);
    end = omp_get_wtime();
    double t_seq = end - start;

    start = omp_get_wtime();
    argmax_par(v, N, &par_m, &par_idx_m);
    end = omp_get_wtime();
    double t_par = end - start;

    start = omp_get_wtime();
    argmax_recursive(v, N, &rec_m, &rec_idx_m, K);
    end = omp_get_wtime();
    double t_rec = end - start;

    start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        argmax_recursive_tasks(v, N, &task_m, &task_idx_m, K);
    }
    end = omp_get_wtime();
    double t_task = end - start;

    printf("Running argmax with K = %d using %d threads\n", K, nthreads);
    printf("sequential for argmax: m = %.2f, idx_m=%d, time=%.6fs\n", seq_m, seq_idx_m, t_seq);
    printf("parallel for argmax: m = %.2f, idx_m=%d, time=%.6fs\n", par_m, par_idx_m, t_par);
    printf("sequential recursive argmax: m = %.2f, idx_m=%d, time=%.6fs\n", rec_m, rec_idx_m, t_rec);
    printf("parallel recursive argmax: m = %.2f, idx_m=%d, time=%.6fs\n", task_m, task_idx_m, t_task);

    free(v);
    return 0;
}