/* ================================================================================
Intrinsic function wrapper for SVE instruction set
References:
1. ARM Compiler SVE User Guide          : https://developer.arm.com/documentation/100891/0612/sve-overview/introducing-sve
2. ARM C Language Extensions for SVE    : https://developer.arm.com/documentation/100987/latest/
================================================================================ */ 

#ifndef __SVE_INTRIN_WRAPPER_H__
#define __SVE_INTRIN_WRAPPER_H__

#include <math.h>
#include <arm_sve.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef NEWTON_ITER
#define NEWTON_ITER 2   // Two Newton iterations is usually sufficient for rsqrt using double type
#endif

#if !defined(__ARM_FEATURE_SVE)
#error Your processor or compiler does not support SVE instruction set, cannot use this sve_intrin_wrapper.h
#endif

#define PTRUE32B    svptrue_b32()
#define PTRUE64B    svptrue_b64()

#if __ARM_FEATURE_SVE_BITS==0
#warning __ARM_FEATURE_SVE_BITS==0 detected, assuming SVE length == 512 bits
#undef   __ARM_FEATURE_SVE_BITS
#define  __ARM_FEATURE_SVE_BITS 512
#endif

#if __ARM_FEATURE_SVE_BITS==128
#define SIMD_LEN_S  4
#define SIMD_LEN_D  2
#define USE_SVE128
#endif 

#if __ARM_FEATURE_SVE_BITS==256
#define SIMD_LEN_S  8
#define SIMD_LEN_D  4
#define USE_SVE256
#endif 

#if __ARM_FEATURE_SVE_BITS==512
#define SIMD_LEN_S  16
#define SIMD_LEN_D  8
#define USE_SVE512
#endif 

#if __ARM_FEATURE_SVE_BITS==1024
#define SIMD_LEN_S  32
#define SIMD_LEN_D  16
#define USE_SVE1024
#endif 

#if __ARM_FEATURE_SVE_BITS==2048
#define SIMD_LEN_S  64
#define SIMD_LEN_D  32
#define USE_SVE2048
#endif 

typedef svfloat32_t vec_f       __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)));
typedef svfloat64_t vec_d       __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)));
typedef svbool_t    vec_cmp_f   __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)));
typedef svbool_t    vec_cmp_d   __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)));

union vec_f_union
{
    vec_f  v;
    float  f[SIMD_LEN_S];
};

union vec_d_union
{
    vec_d  v;
    double d[SIMD_LEN_D];
};

static inline vec_f vec_zero_s() { return svdup_f32_z(PTRUE32B, 0); }
static inline vec_d vec_zero_d() { return svdup_f64_z(PTRUE64B, 0); }

static inline vec_f vec_set1_s(const float  a)    { return svdup_f32_z(PTRUE32B, a); }
static inline vec_d vec_set1_d(const double a)    { return svdup_f64_z(PTRUE64B, a); }

static inline vec_f vec_bcast_s(float  const *a)  { return svdup_f32_z(PTRUE32B, a[0]); }
static inline vec_d vec_bcast_d(double const *a)  { return svdup_f64_z(PTRUE64B, a[0]); }

static inline vec_f vec_load_s (float  const *a)  { return svld1_f32(PTRUE32B, a);  }
static inline vec_d vec_load_d (double const *a)  { return svld1_f64(PTRUE64B, a);  }

static inline vec_f vec_loadu_s(float  const *a)  { return svld1_f32(PTRUE32B, a); }
static inline vec_d vec_loadu_d(double const *a)  { return svld1_f64(PTRUE64B, a); }

static inline void  vec_store_s (float  *a, const vec_f b)  { svst1_f32(PTRUE32B, a, b);  }
static inline void  vec_store_d (double *a, const vec_d b)  { svst1_f64(PTRUE64B, a, b);  }

static inline void  vec_storeu_s(float  *a, const vec_f b)  { svst1_f32(PTRUE32B, a, b); }
static inline void  vec_storeu_d(double *a, const vec_d b)  { svst1_f64(PTRUE64B, a, b); }

static inline vec_f vec_add_s(const vec_f a, const vec_f b) { return svadd_f32_z(PTRUE32B, a, b); }
static inline vec_d vec_add_d(const vec_d a, const vec_d b) { return svadd_f64_z(PTRUE64B, a, b); }

static inline vec_f vec_sub_s(const vec_f a, const vec_f b) { return svsub_f32_z(PTRUE32B, a, b); }
static inline vec_d vec_sub_d(const vec_d a, const vec_d b) { return svsub_f64_z(PTRUE64B, a, b); }

static inline vec_f vec_mul_s(const vec_f a, const vec_f b) { return svmul_f32_z(PTRUE32B, a, b); }
static inline vec_d vec_mul_d(const vec_d a, const vec_d b) { return svmul_f64_z(PTRUE64B, a, b); }

static inline vec_f vec_div_s(const vec_f a, const vec_f b) { return svdiv_f32_z(PTRUE32B, a, b); }
static inline vec_d vec_div_d(const vec_d a, const vec_d b) { return svdiv_f64_z(PTRUE64B, a, b); }

static inline vec_f vec_abs_s(const vec_f a) { return svabs_f32_z(PTRUE32B, a); }
static inline vec_d vec_abs_d(const vec_d a) { return svabs_f64_z(PTRUE64B, a); }

static inline vec_f vec_sqrt_s(const vec_f a) { return svsqrt_f32_z(PTRUE32B, a); }
static inline vec_d vec_sqrt_d(const vec_d a) { return svsqrt_f64_z(PTRUE64B, a); }

static inline vec_f vec_fmadd_s (const vec_f a, const vec_f b, const vec_f c) { return svmad_f32_z(PTRUE32B, a, b, c);  }
static inline vec_d vec_fmadd_d (const vec_d a, const vec_d b, const vec_d c) { return svmad_f64_z(PTRUE64B, a, b, c);  }

static inline vec_f vec_fnmadd_s(const vec_f a, const vec_f b, const vec_f c) { return svmsb_f32_z(PTRUE32B, a, b, c);  }
static inline vec_d vec_fnmadd_d(const vec_d a, const vec_d b, const vec_d c) { return svmsb_f64_z(PTRUE64B, a, b, c);  }

static inline vec_f vec_fmsub_s (const vec_f a, const vec_f b, const vec_f c) { return svnmsb_f32_z(PTRUE32B, a, b, c); }
static inline vec_d vec_fmsub_d (const vec_d a, const vec_d b, const vec_d c) { return svnmsb_f64_z(PTRUE64B, a, b, c); }

static inline vec_f vec_max_s(const vec_f a, const vec_f b) { return svmax_f32_z(PTRUE32B, a, b); }
static inline vec_d vec_max_d(const vec_d a, const vec_d b) { return svmax_f64_z(PTRUE64B, a, b); }

static inline vec_f vec_min_s(const vec_f a, const vec_f b) { return svmin_f32_z(PTRUE32B, a, b); }
static inline vec_d vec_min_d(const vec_d a, const vec_d b) { return svmin_f64_z(PTRUE64B, a, b); }

static inline vec_cmp_f vec_cmp_eq_s(const vec_f a, const vec_f b) { return svcmpeq_f32(PTRUE32B, a, b);  }
static inline vec_cmp_d vec_cmp_eq_d(const vec_d a, const vec_d b) { return svcmpeq_f64(PTRUE64B, a, b);  }

static inline vec_cmp_f vec_cmp_ne_s(const vec_f a, const vec_f b) { return svcmpne_f32(PTRUE32B, a, b); }
static inline vec_cmp_d vec_cmp_ne_d(const vec_d a, const vec_d b) { return svcmpne_f64(PTRUE64B, a, b); }

static inline vec_cmp_f vec_cmp_lt_s(const vec_f a, const vec_f b) { return svcmplt_f32(PTRUE32B, a, b); }
static inline vec_cmp_d vec_cmp_lt_d(const vec_d a, const vec_d b) { return svcmplt_f64(PTRUE64B, a, b); }

static inline vec_cmp_f vec_cmp_le_s(const vec_f a, const vec_f b) { return svcmple_f32(PTRUE32B, a, b); }
static inline vec_cmp_d vec_cmp_le_d(const vec_d a, const vec_d b) { return svcmple_f64(PTRUE64B, a, b); }

static inline vec_cmp_f vec_cmp_gt_s(const vec_f a, const vec_f b) { return svcmpgt_f32(PTRUE32B, a, b); }
static inline vec_cmp_d vec_cmp_gt_d(const vec_d a, const vec_d b) { return svcmpgt_f64(PTRUE64B, a, b); }

static inline vec_cmp_f vec_cmp_ge_s(const vec_f a, const vec_f b) { return svcmpge_f32(PTRUE32B, a, b); }
static inline vec_cmp_d vec_cmp_ge_d(const vec_d a, const vec_d b) { return svcmpge_f64(PTRUE64B, a, b); }

static inline vec_f vec_blend_s(const vec_f a, const vec_f b, const vec_cmp_f mask) { return svsel_f32(mask, b, a); }
static inline vec_d vec_blend_d(const vec_d a, const vec_d b, const vec_cmp_d mask) { return svsel_f64(mask, b, a); }

static inline float  vec_reduce_add_s(const vec_f a) { return svaddv_f32(PTRUE32B, a); }
static inline double vec_reduce_add_d(const vec_d a) { return svaddv_f64(PTRUE64B, a); }

static inline vec_f vec_arsqrt_s(const vec_f a)
{
    vec_f zero  = vec_zero_s();
    vec_f rsqrt = svrsqrte_f32(a);
    vec_cmp_f cmp0 = vec_cmp_eq_s(a, zero);
    return vec_blend_s(rsqrt, zero, cmp0);
}
static inline vec_d vec_arsqrt_d(const vec_d a)
{ 
    vec_d zero  = vec_zero_d();
    vec_d rsqrt = svrsqrte_f64(a);
    vec_cmp_d cmp0 = vec_cmp_eq_d(a, zero);
    return vec_blend_d(rsqrt, zero, cmp0);
}

#ifdef USE_SLEEF
// TODO: add log, exp, pow, sin, cos, erf functions
#else
static inline vec_f vec_log_s(vec_f x)
{
    int i;
    union vec_f_union u = {x}, res;
    for (i = 0; i < SIMD_LEN_S; i++) res.f[i] = logf(u.f[i]);
    return res.v;
}
static inline vec_d vec_log_d(vec_d x)
{
    int i;
    union vec_d_union u = {x}, res;
    for (i = 0; i < SIMD_LEN_D; i++) res.d[i] = log(u.d[i]);
    return res.v;
}

static inline vec_f vec_exp_s(vec_f x)
{
    int i;
    union vec_f_union u = {x}, res;
    for (i = 0; i < SIMD_LEN_S; i++) res.f[i] = expf(u.f[i]);
    return res.v;
}
static inline vec_d vec_exp_d(vec_d x)
{
    int i;
    union vec_d_union u = {x}, res;
    for (i = 0; i < SIMD_LEN_D; i++) res.d[i] = exp(u.d[i]);
    return res.v;
}

static inline vec_f vec_pow_s(vec_f a, vec_f b)
{
    int i;
    union vec_f_union ua = {a}, ub = {b}, res;
    for (i = 0; i < SIMD_LEN_S; i++) res.f[i] = powf(ua.f[i], ub.f[i]);
    return res.v;
}
static inline vec_d vec_pow_d(vec_d a, vec_d b)
{
    int i;
    union vec_d_union ua = {a}, ub = {b}, res;
    for (i = 0; i < SIMD_LEN_D; i++) res.d[i] = pow(ua.d[i], ub.d[i]);
    return res.v;
}

static inline vec_f vec_sin_s(vec_f a)
{
    int i;
    union vec_f_union ua = {a}, res;
    for (i = 0; i < SIMD_LEN_S; i++) res.f[i] = sinf(ua.f[i]);
    return res.v;
}
static inline vec_d vec_sin_d(vec_d a)
{
    int i;
    union vec_d_union ua = {a}, res;
    for (i = 0; i < SIMD_LEN_D; i++) res.d[i] = sin(ua.d[i]);
    return res.v;
}

static inline vec_f vec_cos_s(vec_f a)
{
    int i;
    union vec_f_union ua = {a}, res;
    for (i = 0; i < SIMD_LEN_S; i++) res.f[i] = cosf(ua.f[i]);
    return res.v;
}
static inline vec_d vec_cos_d(vec_d a)
{
    int i;
    union vec_d_union ua = {a}, res;
    for (i = 0; i < SIMD_LEN_D; i++) res.d[i] = cos(ua.d[i]);
    return res.v;
}

static inline vec_f vec_erf_s(vec_f a)
{
    int i;
    union vec_f_union ua = {a}, res;
    for (i = 0; i < SIMD_LEN_S; i++) res.f[i] = erff(ua.f[i]);
    return res.v;
}
static inline vec_d vec_erf_d(vec_d a)
{
    int i;
    union vec_d_union ua = {a}, res;
    for (i = 0; i < SIMD_LEN_D; i++) res.d[i] = erf(ua.d[i]);
    return res.v;
}

static inline vec_f vec_log2_s (const vec_f a) { return vec_div_s(vec_log_s(a), vec_set1_s(M_LN2));  }
static inline vec_d vec_log2_d (const vec_d a) { return vec_div_d(vec_log_d(a), vec_set1_d(M_LN2));  }

static inline vec_f vec_log10_s(const vec_f a) { return vec_div_s(vec_log_s(a), vec_set1_s(M_LN10)); }
static inline vec_d vec_log10_d(const vec_d a) { return vec_div_d(vec_log_d(a), vec_set1_d(M_LN10)); }

static inline vec_f vec_exp2_s (const vec_f a) { return vec_exp_s(vec_mul_s(a, vec_set1_s(M_LN2)));  }
static inline vec_d vec_exp2_d (const vec_d a) { return vec_exp_d(vec_mul_d(a, vec_set1_d(M_LN2)));  }

static inline vec_f vec_exp10_s(const vec_f a) { return vec_exp_s(vec_mul_s(a, vec_set1_s(M_LN10))); }
static inline vec_d vec_exp10_d(const vec_d a) { return vec_exp_d(vec_mul_d(a, vec_set1_d(M_LN10))); }
#endif  // End of "#ifdef USE_SLEEF"

static inline vec_f vec_frsqrt_pf_s() { return vec_set1_s(1); }
static inline vec_d vec_frsqrt_pf_d() { return vec_set1_d(1); }

#define RSQRT_REFINE_F32(pg, rsqrt_target, rsqrt_iter, rsqrt_work)  \
    do  \
    {   \
        rsqrt_work = svmul_f32_z(pg, rsqrt_target, rsqrt_iter); \
        rsqrt_work = svrsqrts_f32(rsqrt_work, rsqrt_iter);      \
        rsqrt_iter = svmul_f32_z(pg, rsqrt_work, rsqrt_iter);   \
    } while (0)

#define RSQRT_REFINE_F64(pg, rsqrt_target, rsqrt_iter, rsqrt_work)  \
    do  \
    {   \
        rsqrt_work = svmul_f64_z(pg, rsqrt_target, rsqrt_iter); \
        rsqrt_work = svrsqrts_f64(rsqrt_work, rsqrt_iter);      \
        rsqrt_iter = svmul_f64_z(pg, rsqrt_work, rsqrt_iter);   \
    } while (0)

static inline vec_f vec_frsqrt_s(const vec_f a)
{
    vec_f rsqrt = vec_arsqrt_s(a);
    vec_f rsqrt_work;
    #if NEWTON_ITER >= 1
    RSQRT_REFINE_F32(PTRUE32B, a, rsqrt, rsqrt_work);
    #endif
    #if NEWTON_ITER >= 2
    RSQRT_REFINE_F32(PTRUE32B, a, rsqrt, rsqrt_work);
    #endif
    #if NEWTON_ITER >= 3
    RSQRT_REFINE_F32(PTRUE32B, a, rsqrt, rsqrt_work);
    #endif
    return rsqrt;
}
static inline vec_d vec_frsqrt_d(const vec_d a)
{
    vec_d rsqrt = vec_arsqrt_d(a);
    vec_d rsqrt_work;
    #if NEWTON_ITER >= 1
    RSQRT_REFINE_F64(PTRUE64B, a, rsqrt, rsqrt_work);
    #endif
    #if NEWTON_ITER >= 2
    RSQRT_REFINE_F64(PTRUE64B, a, rsqrt, rsqrt_work);
    #endif
    #if NEWTON_ITER >= 3
    RSQRT_REFINE_F64(PTRUE64B, a, rsqrt, rsqrt_work);
    #endif
    return rsqrt;
}

#ifdef __cplusplus
}
#endif

#endif
