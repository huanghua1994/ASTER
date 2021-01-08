/* ================================================================================
Fall-back implementation of vector wrapper functions, simulated by for loop
References:
1. Fast inverse square root : https://en.wikipedia.org/wiki/Fast_inverse_square_root
2. Fast inverse square root : https://github.com/dmalhotra/pvfmm/blob/develop/include/intrin_wrapper.hpp
================================================================================ */ 

#ifndef __NO_INTRIN_WRAPPER_H__
#define __NO_INTRIN_WRAPPER_H__

#include <math.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SIMD_LEN_S  16
#define SIMD_LEN_D  8

struct vec16f
{
    float f[SIMD_LEN_S];
} __attribute__ ((aligned(64)));
typedef struct vec16f vec_f;

struct vec8d
{
    double d[SIMD_LEN_D];
} __attribute__ ((aligned(64)));
typedef struct vec8d vec_d;

struct vec16i
{
    int i[SIMD_LEN_S];
} __attribute__ ((aligned(64)));
typedef struct vec16i vec_cmp_f;

struct vec8i
{
    int i[SIMD_LEN_D];
} __attribute__ ((aligned(64)));
typedef struct vec8i vec_cmp_d;

static inline vec_f vec_zero_s()
{
    vec_f res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_S; i++) res.f[i] = 0.0;
    return res;
}
static inline vec_d vec_zero_d()
{
    vec_d res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.d[i] = 0.0;
    return res;
}

static inline vec_f vec_set1_s(const float  a)
{
    vec_f res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_S; i++) res.f[i] = a;
    return res;
}
static inline vec_d vec_set1_d(const double a)
{
    vec_d res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.d[i] = a;
    return res;
}

static inline vec_f vec_bcast_s(float  const *a)
{
    vec_f res;
    float a0 = a[0];
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_S; i++) res.f[i] = a0;
    return res;
}
static inline vec_d vec_bcast_d(double const *a)
{
    vec_d res;
    double a0 = a[0];
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.d[i] = a0;
    return res;
}

static inline vec_f vec_load_s (float  const *a)
{
    vec_f res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_S; i++) res.f[i] = a[i];
    return res;
}
static inline vec_d vec_load_d (double const *a)
{
    vec_d res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.d[i] = a[i];
    return res;
}

static inline vec_f vec_loadu_s(float  const *a)  { return vec_load_s(a); }
static inline vec_d vec_loadu_d(double const *a)  { return vec_load_d(a); }

static inline void vec_store_s (float  *a, const vec_f b)
{
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_S; i++) a[i] = b.f[i];
}
static inline void vec_store_d (double *a, const vec_d b)
{
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) a[i] = b.d[i];
}

static inline void vec_storeu_s(float  *a, const vec_f b) { vec_store_s(a, b); }
static inline void vec_storeu_d(double *a, const vec_d b) { vec_store_d(a, b); }

static inline vec_f vec_add_s(const vec_f a, const vec_f b)
{
    vec_f res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_S; i++) res.f[i] = a.f[i] + b.f[i];
    return res;
}
static inline vec_d vec_add_d(const vec_d a, const vec_d b)
{
    vec_d res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.d[i] = a.d[i] + b.d[i];
    return res;
}

static inline vec_f vec_sub_s(const vec_f a, const vec_f b)
{
    vec_f res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_S; i++) res.f[i] = a.f[i] - b.f[i];
    return res;
}
static inline vec_d vec_sub_d(const vec_d a, const vec_d b)
{
    vec_d res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.d[i] = a.d[i] - b.d[i];
    return res;
}

static inline vec_f vec_mul_s(const vec_f a, const vec_f b)
{
    vec_f res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_S; i++) res.f[i] = a.f[i] * b.f[i];
    return res;
}
static inline vec_d vec_mul_d(const vec_d a, const vec_d b)
{
    vec_d res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.d[i] = a.d[i] * b.d[i];
    return res;
}

static inline vec_f vec_div_s(const vec_f a, const vec_f b)
{
    vec_f res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_S; i++) res.f[i] = a.f[i] / b.f[i];
    return res;
}
static inline vec_d vec_div_d(const vec_d a, const vec_d b)
{
    vec_d res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.d[i] = a.d[i] / b.d[i];
    return res;
}

static inline vec_f vec_abs_s(const vec_f a)
{
    vec_f res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_S; i++) res.f[i] = fabsf(a.f[i]);
    return res;
}
static inline vec_d vec_abs_d(const vec_d a)
{
    vec_d res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.d[i] = fabs(a.d[i]);
    return res;
}

static inline vec_f vec_sqrt_s(const vec_f a)
{
    vec_f res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_S; i++) res.f[i] = sqrtf(a.f[i]);
    return res;
}
static inline vec_d vec_sqrt_d(const vec_d a)
{
    vec_d res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.d[i] = sqrt(a.d[i]);
    return res;
}

static inline vec_f vec_fmadd_s(const vec_f a, const vec_f b, const vec_f c)
{
    vec_f res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_S; i++) res.f[i] = a.f[i] * b.f[i] + c.f[i];
    return res;
}
static inline vec_d vec_fmadd_d(const vec_d a, const vec_d b, const vec_d c)
{
    vec_d res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.d[i] = a.d[i] * b.d[i] + c.d[i];
    return res;
}

static inline vec_f vec_fnmadd_s(const vec_f a, const vec_f b, const vec_f c)
{
    vec_f res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_S; i++) res.f[i] = -(a.f[i] * b.f[i]) + c.f[i];
    return res;
}
static inline vec_d vec_fnmadd_d(const vec_d a, const vec_d b, const vec_d c)
{
    vec_d res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.d[i] = -(a.d[i] * b.d[i]) + c.d[i];
    return res;
}

static inline vec_f vec_fmsub_s (const vec_f a, const vec_f b, const vec_f c)
{
    vec_f res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_S; i++) res.f[i] = a.f[i] * b.f[i] - c.f[i];
    return res;
}
static inline vec_d vec_fmsub_d (const vec_d a, const vec_d b, const vec_d c)
{
    vec_d res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.d[i] = a.d[i] * b.d[i] - c.d[i];
    return res;
}

static inline vec_f vec_max_s(const vec_f a, const vec_f b)
{
    vec_f res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_S; i++) res.f[i] = (a.f[i] > b.f[i]) ? a.f[i] : b.f[i];
    return res;
}
static inline vec_d vec_max_d(const vec_d a, const vec_d b)
{
    vec_d res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.d[i] = (a.d[i] > b.d[i]) ? a.d[i] : b.d[i];
    return res;
}

static inline vec_f vec_min_s(const vec_f a, const vec_f b)
{
    vec_f res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_S; i++) res.f[i] = (a.f[i] < b.f[i]) ? a.f[i] : b.f[i];
    return res;
}
static inline vec_d vec_min_d(const vec_d a, const vec_d b)
{
    vec_d res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.d[i] = (a.d[i] < b.d[i]) ? a.d[i] : b.d[i];
    return res;
}

static inline vec_cmp_f vec_cmp_eq_s (const vec_f a, const vec_f b)
{
    vec_cmp_f res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_S; i++) res.i[i] = (a.f[i] == b.f[i]);
    return res;
}
static inline vec_cmp_d vec_cmp_eq_d (const vec_d a, const vec_d b)
{
    vec_cmp_d res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.i[i] = (a.d[i] == b.d[i]);
    return res;
}

static inline vec_cmp_f vec_cmp_neq_s(const vec_f a, const vec_f b)
{
    vec_cmp_f res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_S; i++) res.i[i] = (a.f[i] != b.f[i]);
    return res;
}
static inline vec_cmp_d vec_cmp_neq_d(const vec_d a, const vec_d b)
{
    vec_cmp_d res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.i[i] = (a.d[i] != b.d[i]);
    return res;
}

static inline vec_cmp_f vec_cmp_lt_s (const vec_f a, const vec_f b)
{
    vec_cmp_f res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_S; i++) res.i[i] = (a.f[i] < b.f[i]);
    return res;
}
static inline vec_cmp_d vec_cmp_lt_d (const vec_d a, const vec_d b)
{
    vec_cmp_d res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.i[i] = (a.d[i] < b.d[i]);
    return res;
}

static inline vec_cmp_f vec_cmp_le_s (const vec_f a, const vec_f b)
{
    vec_cmp_f res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_S; i++) res.i[i] = (a.f[i] <= b.f[i]);
    return res;
}
static inline vec_cmp_d vec_cmp_le_d (const vec_d a, const vec_d b)
{
    vec_cmp_d res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.i[i] = (a.d[i] <= b.d[i]);
    return res;
}

static inline vec_cmp_f vec_cmp_gt_s (const vec_f a, const vec_f b)
{
    vec_cmp_f res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_S; i++) res.i[i] = (a.f[i] > b.f[i]);
    return res;
}
static inline vec_cmp_d vec_cmp_gt_d (const vec_d a, const vec_d b)
{
    vec_cmp_d res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.i[i] = (a.d[i] > b.d[i]);
    return res;
}

static inline vec_cmp_f vec_cmp_ge_s (const vec_f a, const vec_f b)
{
    vec_cmp_f res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_S; i++) res.i[i] = (a.f[i] >= b.f[i]);
    return res;
}
static inline vec_cmp_d vec_cmp_ge_d (const vec_d a, const vec_d b)
{
    vec_cmp_d res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.i[i] = (a.d[i] >= b.d[i]);
    return res;
}

static inline vec_f vec_blend_s(const vec_f a, const vec_f b, const vec_cmp_f mask)
{
    vec_f res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_S; i++) res.f[i] = (mask.i[i]) ? b.f[i] : a.f[i];
    return res;
}
static inline vec_d vec_blend_d(const vec_d a, const vec_d b, const vec_cmp_d mask)
{
    vec_d res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.d[i] = (mask.i[i]) ? b.d[i] : a.d[i];
    return res;
}

static inline float  vec_reduce_add_s(const vec_f a)
{
    float res = 0.0;
    #pragma omp simd reduction(+:res)
    for (int i = 0; i < SIMD_LEN_S; i++) res += a.f[i];
    return res;
}
static inline double vec_reduce_add_d(const vec_d a)
{
    double res = 0.0;
    #pragma omp simd reduction(+:res)
    for (int i = 0; i < SIMD_LEN_D; i++) res += a.d[i];
    return res;
}

static inline vec_f vec_log_s  (const vec_f a)
{
    vec_f res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_S; i++) res.f[i] = logf(a.f[i]);
    return res;
}
static inline vec_d vec_log_d  (const vec_d a)
{
    vec_d res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.d[i] = log(a.d[i]);
    return res;
}

static inline vec_f vec_log2_s (const vec_f a)
{
    vec_f res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_S; i++) res.f[i] = log2f(a.f[i]);
    return res;
}
static inline vec_d vec_log2_d (const vec_d a)
{
    vec_d res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.d[i] = log2(a.d[i]);
    return res;
}

static inline vec_f vec_log10_s(const vec_f a)
{
    vec_f res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_S; i++) res.f[i] = log10f(a.f[i]);
    return res;
}
static inline vec_d vec_log10_d(const vec_d a)
{
    vec_d res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.d[i] = log10(a.d[i]);
    return res;
}

static inline vec_f vec_exp_s  (const vec_f a)
{
    vec_f res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_S; i++) res.f[i] = expf(a.f[i]);
    return res;
}
static inline vec_d vec_exp_d  (const vec_d a)
{
    vec_d res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.d[i] = exp(a.d[i]);
    return res;
}

static inline vec_f vec_exp2_s (const vec_f a)
{
    vec_f res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_S; i++) res.f[i] = exp2f(a.f[i]);
    return res;
}
static inline vec_d vec_exp2_d (const vec_d a)
{
    vec_d res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.d[i] = exp2(a.d[i]);
    return res;
}

static inline vec_f vec_exp10_s(const vec_f a) { return vec_exp_s(vec_mul_s(a, vec_set1_s(M_LN10))); }
static inline vec_d vec_exp10_d(const vec_d a) { return vec_exp_d(vec_mul_d(a, vec_set1_d(M_LN10))); }

static inline vec_f vec_pow_s  (const vec_f a, const vec_f b)
{
    vec_f res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_S; i++) res.f[i] = powf(a.f[i], b.f[i]);
    return res;
}
static inline vec_d vec_pow_d  (const vec_d a, const vec_d b)
{
    vec_d res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.d[i] = pow(a.d[i], b.d[i]);
    return res;
}

static inline vec_f vec_sin_s  (const vec_f a)
{
    vec_f res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.f[i] = sinf(a.f[i]);
    return res;
}
static inline vec_d vec_sin_d  (const vec_d a)
{
    vec_d res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.d[i] = sin(a.d[i]);
    return res;
}

static inline vec_f vec_cos_s  (const vec_f a)
{
    vec_f res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.f[i] = cosf(a.f[i]);
    return res;
}
static inline vec_d vec_cos_d  (const vec_d a)
{
    vec_d res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.d[i] = cos(a.d[i]);
    return res;
}

static inline vec_f vec_erf_s  (const vec_f a)
{
    vec_f res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.f[i] = erff(a.f[i]);
    return res;
}
static inline vec_d vec_erf_d  (const vec_d a)
{
    vec_d res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) res.d[i] = erf(a.d[i]);
    return res;
}

//#define USE_FAST_RSQRT

#ifndef NEWTON_ITER
#define NEWTON_ITER 3
#endif

#ifdef USE_FAST_RSQRT
static inline vec_f vec_arsqrt_s(const vec_f r2)
{
    vec_f res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_S; i++)
    {
        uint32_t ii;
        float x2, y;
        x2 = r2.f[i] * 0.5f;
        y  = r2.f[i];
        ii = *(uint32_t *) &y;
        ii = 0x5f375a86 - (ii >> 1);
        y  = *(float *) &ii;
        y  = y * (1.5f - (x2 * y * y));
        res.f[i] = (r2.f[i] == 0.0f) ? 0.0f : y;
    }
    return res;
}
static inline vec_d vec_arsqrt_d(const vec_d r2)
{
    vec_d res;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++)
    {
        uint64_t ii;
        double x2, y;
        x2 = r2.d[i] * 0.5;
        y  = r2.d[i];
        ii = *(uint64_t *) &y;
        ii = 0x5fe6eb50c7b537a9 - (ii >> 1);
        y  = *(double *) &ii;
        y  = y * (1.5 - (x2 * y * y));
        res.d[i] = (r2.d[i] == 0.0) ? 0.0 : y;
    }
    return res;
}

// Newton iteration step for reverse square root, rsqrt' = 0.5 * rsqrt * (C - r2 * rsqrt^2),
// 0.5 is ignored here and need to be adjusted outside. 
static inline vec_f vec_rsqrt_ntit_s(const vec_f r2, vec_f rsqrt, const float  C_)
{
    vec_f C  = vec_set1_s(C_);
    vec_f t1 = vec_mul_s(rsqrt, rsqrt);
    vec_f t2 = vec_fnmadd_s(r2, t1, C);
    return vec_mul_s(rsqrt, t2);
}
static inline vec_d vec_rsqrt_ntit_d(const vec_d r2, vec_d rsqrt, const double C_)
{
    vec_d C  = vec_set1_d(C_);
    vec_d t1 = vec_mul_d(rsqrt, rsqrt);
    vec_d t2 = vec_fnmadd_d(r2, t1, C);
    return vec_mul_d(rsqrt, t2);
}
#endif  // ifdef USE_FAST_RSQRT

static inline vec_f vec_frsqrt_pf_s()
{
    #if defined(USE_FAST_RSQRT)
    float newton_pf = 1.0;
    for (int i = 0; i < NEWTON_ITER; i++)
        newton_pf = 2.0 * newton_pf * newton_pf * newton_pf;
    newton_pf = 1.0 / newton_pf;
    return vec_set1_s(newton_pf);
    #else
    return vec_set1_s(1.0);
    #endif
}
static inline vec_d vec_frsqrt_pf_d()
{
    #if defined(USE_FAST_RSQRT)
    double newton_pf = 1.0;
    for (int i = 0; i < NEWTON_ITER; i++)
        newton_pf = 2.0 * newton_pf * newton_pf * newton_pf;
    newton_pf = 1.0 / newton_pf;
    return vec_set1_d(newton_pf);
    #else
    return vec_set1_d(1.0);
    #endif
}

static inline vec_f vec_frsqrt_s(const vec_f r2)
{
    #if defined(USE_FAST_RSQRT)
    vec_f rsqrt = vec_arsqrt_s(r2);
    #if NEWTON_ITER >= 1
    rsqrt = vec_rsqrt_ntit_s(r2, rsqrt, 3);
    #endif
    #if NEWTON_ITER >= 2
    rsqrt = vec_rsqrt_ntit_s(r2, rsqrt, 12);
    #endif
    #if NEWTON_ITER >= 3
    rsqrt = vec_rsqrt_ntit_s(r2, rsqrt, 768);
    #endif
    return rsqrt;
    #else
    vec_f rsqrt;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_S; i++) rsqrt.f[i] = (r2.f[i] == 0.0) ? 0.0 : (1.0 / sqrtf(r2.f[i]));
    #endif
    return rsqrt;
}
static inline vec_d vec_frsqrt_d(const vec_d r2)
{
    #if defined(USE_FAST_RSQRT)
    vec_d rsqrt = vec_arsqrt_d(r2);
    #if NEWTON_ITER >= 1
    rsqrt = vec_rsqrt_ntit_d(r2, rsqrt, 3);
    #endif
    #if NEWTON_ITER >= 2
    rsqrt = vec_rsqrt_ntit_d(r2, rsqrt, 12);
    #endif
    #if NEWTON_ITER >= 3
    rsqrt = vec_rsqrt_ntit_d(r2, rsqrt, 768);
    #endif
    #else
    vec_d rsqrt;
    #pragma omp simd
    for (int i = 0; i < SIMD_LEN_D; i++) rsqrt.d[i] = (r2.d[i] == 0.0) ? 0.0 : (1.0 / sqrt(r2.d[i]));
    #endif
    return rsqrt;
}

#ifdef __cplusplus
}
#endif

#endif  // End of header file 
