#include <stdio.h>
#include <stdlib.h>
#include "cholesky.h"

int main(int argc, char* argv[]) {
    int n;
    
    if(argc == 2) {
        n = atoi(argv[1]);
    } else {
        fprintf(stderr, "Wrong number of paramethers\nUsage:\n\t%s N (for NxN matrix)\n",argv[0]);
        return -1;
    }
    
    printf("Sequential Cholesky\n");
    cholesky(n);
    printf("\nOpenMP Cholesky\n");
    cholesky_openmp(n);
    
    return 0;
}
