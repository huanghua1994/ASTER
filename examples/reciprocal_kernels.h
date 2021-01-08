#ifndef __RECIPROCAL_KERNELS_H__
#define __RECIPROCAL_KERNELS_H__

#include <math.h>
#include "aster.h"

// Pointer to function that performs kernel matrix matvec using given sets of 
// points and given input vector. The kernel function must be symmetric.
// Input parameters:
//   coord0 : Matrix, size 3-by-ld0, coordinates of the 1st point set
//   ld0    : Leading dimension of coord0, should be >= n0
//   n0     : Number of points in coord0 (each column in coord0 is a coordinate)
//   coord1 : Matrix, size 3-by-ld1, coordinates of the 2nd point set
//   ld1    : Leading dimension of coord1, should be >= n1
//   n1     : Number of points in coord1 (each column in coord0 is a coordinate)
//   x_in   : Vector, size >= n1, will be left multiplied by kernel_matrix(coord0, coord1)
// Output parameter:
//   x_out : Vector, size >= n0, x_out += kernel_matrix(coord0, coord1) * x_in
typedef void (*kernel_matvec_fptr) (
    const double *coord0, const int ld0, const int n0,
    const double *coord1, const int ld1, const int n1,
    const double *x_in, double *x_out
);

static void reciprocal_matvec_std(
    const double *coord0, const int ld0, const int n0,
    const double *coord1, const int ld1, const int n1,
    const double *x_in, double *x_out
)
{
    const double *x0 = coord0 + ld0 * 0;
    const double *y0 = coord0 + ld0 * 1;
    const double *z0 = coord0 + ld0 * 2;
    const double *x1 = coord1 + ld1 * 0;
    const double *y1 = coord1 + ld1 * 1;
    const double *z1 = coord1 + ld1 * 2;
    for (int i = 0; i < n0; i++)
    {
        const double x0_i = x0[i];
        const double y0_i = y0[i];
        const double z0_i = z0[i];
        double sum = 0.0;
        #pragma omp simd
        for (int j = 0; j < n1; j++)
        {
            double dx = x0_i - x1[j];
            double dy = y0_i - y1[j];
            double dz = z0_i - z1[j];
            double r2 = dx * dx + dy * dy + dz * dz;
            double res = (r2 == 0.0) ? 0.0 : (x_in[j] / sqrt(r2));
            sum += res;
        }
        x_out[i] += sum;
    }
}

static void reciprocal_matvec_intrin(
    const double *coord0, const int ld0, const int n0,
    const double *coord1, const int ld1, const int n1,
    const double *x_in, double *x_out
)
{
    const double *x0 = coord0 + ld0 * 0;
    const double *y0 = coord0 + ld0 * 1;
    const double *z0 = coord0 + ld0 * 2;
    const double *x1 = coord1 + ld1 * 0;
    const double *y1 = coord1 + ld1 * 1;
    const double *z1 = coord1 + ld1 * 2;
    vec_d frsqrt_pf = vec_frsqrt_pf_d();
    int i = 0;
    const int blk_size = 1024;
    for (int j_sidx = 0; j_sidx < n1; j_sidx += blk_size)
    {
        int j_eidx = (j_sidx + blk_size > n1) ? n1 : (j_sidx + blk_size);
        for (i = 0; i <= n0 - SIMD_LEN_D; i += SIMD_LEN_D)
        {
            vec_d tx = vec_loadu_d(x0 + i);
            vec_d ty = vec_loadu_d(y0 + i);
            vec_d tz = vec_loadu_d(z0 + i);
            vec_d tv = vec_zero_d();
            for (int j = j_sidx; j < j_eidx; j++)
            {
                vec_d dx = vec_sub_d(tx, vec_bcast_d(x1 + j));
                vec_d dy = vec_sub_d(ty, vec_bcast_d(y1 + j));
                vec_d dz = vec_sub_d(tz, vec_bcast_d(z1 + j));
                
                vec_d r2 = vec_mul_d(dx, dx);
                r2 = vec_fmadd_d(dy, dy, r2);
                r2 = vec_fmadd_d(dz, dz, r2);
                
                vec_d sv = vec_mul_d(vec_bcast_d(x_in + j), frsqrt_pf);
                vec_d rinv = vec_frsqrt_d(r2);
                tv = vec_fmadd_d(rinv, sv, tv);
            }
            vec_d outval = vec_loadu_d(x_out + i);
            vec_storeu_d(x_out + i, vec_add_d(outval, tv));
        }
    }
    reciprocal_matvec_std(
        coord0 + i, ld0, n0 - i,
        coord1, ld1, n1,
        x_in, x_out + i
    );
}

#endif
