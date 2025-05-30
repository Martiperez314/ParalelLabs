#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <openacc.h>

int main(int argc, char* argv[]) {
    int N = atoi(argv[1]);
    int write_output = atoi(argv[2]);

    float *x = malloc(N * sizeof(float));
    float *y = malloc(N * sizeof(float));
    float *z = malloc(N * sizeof(float));

    // Initialisierung
    for (int i = 0; i < N; i++) {
        x[i] = y[i] = z[i] = 0.0f;
    }

    #pragma acc data copyin(x[0:N], y[0:N]) copyout(z[0:N])
    {
        #pragma acc parallel loop
        for (int i = 0; i < N; i++) {
            z[i] = x[i] + y[i];
        }
    }

    if (write_output) {
        FILE *f = fopen("particles_output.csv", "w");
        fprintf(f, "x,y,z\n");
        for (int i = 0; i < N; i++) {
            fprintf(f, "%f,%f,%f\n", x[i], y[i], z[i]);
        }
        fclose(f);
    }

    free(x);
    free(y);
    free(z);
    return 0;
}
