#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include "cholesky.h"



void cholesky_openmp(int n) {
    int i, j, k;
    double** A;
    double** L;
    double** U;
    double** B;
    double tmp;
    double start, end;
    int cnt = 0;

    // 1. Matrix initialization
    start = omp_get_wtime();
    A = (double **)malloc(n * sizeof(double *)); 
    L = (double **)malloc(n * sizeof(double *)); 
    U = (double **)malloc(n * sizeof(double *)); 
    B = (double **)malloc(n * sizeof(double *)); 
    
    #pragma omp parallel for
    for(i=0; i<n; i++) {
        A[i] = (double *)malloc(n * sizeof(double)); 
        L[i] = (double *)malloc(n * sizeof(double)); 
        U[i] = (double *)malloc(n * sizeof(double)); 
        B[i] = (double *)malloc(n * sizeof(double)); 
    }

    srand(time(NULL));

    #pragma omp parallel for private(i,j)
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            A[i][j] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }
    
    #pragma omp parallel for private(i,j)
    for (i = 0; i < n; i++) {
        for (j = i; j < n; j++) {
            if (i == j) {
                A[i][j] += n;
            } else {
                double val = ((double) rand() / RAND_MAX) * sqrt(n);
                A[i][j] += val;
                A[j][i] = A[i][j];
            }
        }
    }

    #pragma omp parallel for collapse(2)
    for(i=0; i < n; i++) {
        for(j=0; j < n; j++) {
            L[i][j] = 0.0;
            U[i][j] = 0.0;
        }
    }
    end = omp_get_wtime();
    printf("Initialization: %f\n", end-start);


    // 2. Cholesky factorization
    start = omp_get_wtime();
    for(i = 0; i < n; i++) {
        tmp = 0.0;
        for(k = 0; k < i; k++) {
            tmp += U[k][i] * U[k][i];
        }
        U[i][i] = sqrt(A[i][i] - tmp);

        #pragma omp parallel for private(k, tmp)
        for(j = i + 1; j < n; j++) {
            tmp = 0.0;
            for(k = 0; k < i; k++) {
                tmp += U[k][i] * U[k][j];
            }
            U[i][j] = (A[i][j] - tmp) / U[i][i];
        }
    }
    end = omp_get_wtime();
    printf("Cholesky: %f\n", end-start);


    start = omp_get_wtime();
    #pragma omp parallel for collapse(2)
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            L[i][j] = U[j][i];
        }
    }
    end = omp_get_wtime();
    printf("L=U': %f\n", end-start);


    // 4. Compute B = L * U
    start = omp_get_wtime();
    #pragma omp parallel for private(j,k)
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            B[i][j] = 0.0;
            for (k = 0; k < n; k++) {
                B[i][j] += L[i][k] * U[k][j];
            }
        }
    }
    end = omp_get_wtime();
    printf("B=LU: %f\n", end-start);


    // 5. Check A = B
    start = omp_get_wtime();
    cnt = 0;
    #pragma omp parallel for collapse(2) reduction(+:cnt)
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (fabs(A[i][j] - B[i][j]) > 0.00001) {
                cnt++;
            }
        }
    }
    end = omp_get_wtime();
    if(cnt != 0) {
        printf("Matrices are not equal\n");
    } else {
        printf("Matrices are equal\n");
    }
    printf("A==B?: %d\n", cnt);

    #pragma omp parallel for
    for(i=0; i<n; i++) {
        free(A[i]);
        free(L[i]);
        free(U[i]);
        free(B[i]);
    }
    free(A);
    free(L);
    free(U);
    free(B);
}



void cholesky(int n) {
    int i, j, k;
    double** A;
    double** L;
    double** U;
    double** B;
    double tmp;
    double start, end;
    int cnt = 0;

    // 1. Matrix initialization
    start = omp_get_wtime();
    A = (double **)malloc(n * sizeof(double *)); 
    L = (double **)malloc(n * sizeof(double *)); 
    U = (double **)malloc(n * sizeof(double *)); 
    B = (double **)malloc(n * sizeof(double *)); 
    
    for(i=0; i<n; i++) {
        A[i] = (double *)malloc(n * sizeof(double)); 
        L[i] = (double *)malloc(n * sizeof(double)); 
        U[i] = (double *)malloc(n * sizeof(double)); 
        B[i] = (double *)malloc(n * sizeof(double)); 
    }

    srand(time(NULL));

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            A[i][j] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }
    

    for (i = 0; i < n; i++) {
        for (j = i; j < n; j++) {
            if (i == j) {
                A[i][j] += n;
            } else {
                double val = ((double) rand() / RAND_MAX) * sqrt(n);
                A[i][j] += val;
                A[j][i] = A[i][j];
            }
        }
    }

    for(i=0; i < n; i++) {
        for(j=0; j < n; j++) {
            L[i][j] = 0.0;
            U[i][j] = 0.0;
        }
    }
    end = omp_get_wtime();
    printf("Initialization: %f\n", end-start);


    // 2. Cholesky factorization
    start = omp_get_wtime();
    for(i = 0; i < n; i++) {
        tmp = 0.0;
        for(k = 0; k < i; k++) {
            tmp += U[k][i] * U[k][i];
        }
        U[i][i] = sqrt(A[i][i] - tmp);

        for(j = i + 1; j < n; j++) {
            tmp = 0.0;
            for(k = 0; k < i; k++) {
                tmp += U[k][i] * U[k][j];
            }
            U[i][j] = (A[i][j] - tmp) / U[i][i];
        }
    }
    end = omp_get_wtime();
    printf("Cholesky: %f\n", end-start);


    start = omp_get_wtime();
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            L[i][j] = U[j][i];
        }
    }
    end = omp_get_wtime();
    printf("L=U': %f\n", end-start);


    // 4. Compute B = L * U
    start = omp_get_wtime();
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            B[i][j] = 0.0;
            for (k = 0; k < n; k++) {
                B[i][j] += L[i][k] * U[k][j];
            }
        }
    }
    end = omp_get_wtime();
    printf("B=LU: %f\n", end-start);


    // 5. Check A = B
    start = omp_get_wtime();
    cnt = 0;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (fabs(A[i][j] - B[i][j]) > 0.00001) {
                cnt++;
            }
        }
    }
    end = omp_get_wtime();
    if(cnt != 0) {
        printf("Matrices are not equal\n");
    } else {
        printf("Matrices are equal\n");
    }
    printf("A==B?: %d\n", cnt);

    for(i=0; i<n; i++) {
        free(A[i]);
        free(L[i]);
        free(U[i]);
        free(B[i]);
    }
    free(A);
    free(L);
    free(U);
    free(B);
}

