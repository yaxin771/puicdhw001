// src/matmul_blocked_openmp.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

static inline double* aligned_malloc(size_t n) {
    void *p = NULL;
    posix_memalign(&p, 64, n * sizeof(double));
    return (double*)p;
}

void matmul_blocked_openmp(double *A, double *B, double *C, int N, int Bsize) {
    int i, j, k, ii, jj, kk;
    #pragma omp parallel for private(i,j,k,ii,jj,kk) schedule(dynamic)
    for (ii = 0; ii < N; ii += Bsize) {
        for (jj = 0; jj < N; jj += Bsize) {
            for (kk = 0; kk < N; kk += Bsize) {
                int i_max = (ii + Bsize > N) ? N : ii + Bsize;
                int j_max = (jj + Bsize > N) ? N : jj + Bsize;
                int k_max = (kk + Bsize > N) ? N : kk + Bsize;
                for (i = ii; i < i_max; ++i) {
                    for (k = kk; k < k_max; ++k) {
                        double a = A[i*N + k];
                        for (j = jj; j < j_max; ++j) {
                            C[i*N + j] += a * B[k*N + j];
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s N Bsize\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    int Bsize = atoi(argv[2]);

    double *A = aligned_malloc((size_t)N*N);
    double *B = aligned_malloc((size_t)N*N);
    double *C = aligned_malloc((size_t)N*N);
    srand(0);
    for (size_t i=0;i<(size_t)N*N;i++) { A[i] = drand48(); B[i] = drand48(); C[i] = 0.0; }

    double t0 = omp_get_wtime();
    matmul_blocked_openmp(A,B,C,N,Bsize);
    double t1 = omp_get_wtime();
    double seconds = t1 - t0;
    double gflops = 2.0 * N * N * N / (seconds * 1e9);
    printf("N=%d B=%d time=%.6f s GFLOPS=%.3f\n", N, Bsize, seconds, gflops);

    free(A); free(B); free(C);
    return 0;
}
