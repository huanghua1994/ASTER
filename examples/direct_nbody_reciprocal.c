#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#include "reciprocal_kernels.h"

void test_direct_nbody(
    const int n_src, const double *src_coord, double *src_val, 
    const int n_trg, const double *trg_coord, double *trg_val,
    kernel_matvec_fptr krnl_matvec, const int krnl_dim, int krnl_flops
)
{
    int nthread = omp_get_max_threads();
    
    double krnl_GFLOPS = (double)n_src * (double)n_trg * (double)krnl_flops;
    krnl_GFLOPS /= 1000000000.0;
    
    for (int k = 0; k < 5; k++)
    {
        memset(trg_val, 0, sizeof(double) * n_trg * krnl_dim);
        double st = omp_get_wtime();
        #pragma omp parallel 
        {
            int tid = omp_get_thread_num();
            
            int trg_sidx = tid * n_trg / nthread;
            int trg_eidx = (tid + 1) * n_trg / nthread;
            int n_trg_thread = trg_eidx - trg_sidx;
            krnl_matvec(
                trg_coord + trg_sidx, n_trg, n_trg_thread, 
                src_coord, n_src, n_src, 
                src_val, trg_val + trg_sidx * krnl_dim
            );
        }
        double et = omp_get_wtime();
        double ut = et - st;
        printf("Direct N-Body %2d: %.3lf s, %.2lf GFLOPS\n", k, ut, krnl_GFLOPS / ut);
    }
}

int main(int argc, char **argv)
{
    int n_src = 0, n_trg = 0;
    if (argc >= 2) n_src = atoi(argv[1]);
    if (argc >= 3) n_trg = atoi(argv[2]);
    if (n_src == 0)
    {
        printf("n_src = ");
        scanf("%d", &n_src);
    }
    if (n_trg == 0)
    {
        printf("n_trg = ");
        scanf("%d", &n_trg);
    }
    
    int krnl_dim = 1;
    
    double *src_coord = (double*) malloc(sizeof(double) * n_src * 3);
    double *trg_coord = (double*) malloc(sizeof(double) * n_trg * 3);
    double *src_val   = (double*) malloc(sizeof(double) * n_src * krnl_dim);
    double *trg_val0  = (double*) malloc(sizeof(double) * n_trg * krnl_dim);
    double *trg_val1  = (double*) malloc(sizeof(double) * n_trg * krnl_dim);
    srand48(time(NULL));
    for (int i = 0; i < n_src; i++) 
    {
        src_coord[i + n_src * 0] = drand48();
        src_coord[i + n_src * 1] = drand48();
        src_coord[i + n_src * 2] = drand48();
    }
    for (int i = 0; i < n_src * krnl_dim; i++) src_val[i] = drand48();
    for (int i = 0; i < n_trg; i++) 
    {
        trg_coord[i + n_trg * 0] = drand48();
        trg_coord[i + n_trg * 1] = drand48();
        trg_coord[i + n_trg * 2] = drand48();
        trg_val0[i]  = 0.0;
        trg_val1[i]  = 0.0;
    }
    
    printf("Standard kernel:\n");
    kernel_matvec_fptr ref_matvec = reciprocal_matvec_std;
    test_direct_nbody(
        n_src, src_coord, src_val,
        n_trg, trg_coord, trg_val0, 
        ref_matvec, krnl_dim, 10
    );
    
    printf("Intrinsic kernel:\n");
    kernel_matvec_fptr avx_matvec = reciprocal_matvec_intrin;
    test_direct_nbody(
        n_src, src_coord, src_val, 
        n_trg, trg_coord, trg_val1, 
        avx_matvec, krnl_dim, 10
    );
    
    double ref_l2 = 0.0, err_l2 = 0.0;
    for (int i = 0; i < n_trg * krnl_dim; i++)
    {
        double diff = trg_val0[i] - trg_val1[i];
        ref_l2 += trg_val0[i] * trg_val0[i];
        err_l2 += diff * diff;
    }
    ref_l2 = sqrt(ref_l2);
    err_l2 = sqrt(err_l2);
    printf("Intrinsic kernel result relative L2 error = %e\n", err_l2 / ref_l2);
    
    free(src_coord);
    free(trg_coord);
    free(src_val);
    free(trg_val0);
    free(trg_val1);
    return 0;
}