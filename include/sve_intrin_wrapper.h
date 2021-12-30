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

typedef svfloat32_t vec_s       __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)));
typedef svfloat64_t vec_d       __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)));
typedef svbool_t    vec_cmp_s   __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)));
typedef svbool_t    vec_cmp_d   __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)));

union vec_s_union
{
    vec_s  v;
    float  f[SIMD_LEN_S];
};

union vec_d_union
{
    vec_d  v;
    double d[SIMD_LEN_D];
};

static inline vec_s vec_zero_s() { return svdup_f32_z(PTRUE32B, 0); }
static inline vec_d vec_zero_d() { return svdup_f64_z(PTRUE64B, 0); }

static inline vec_s vec_set1_s(const float  a)    { return svdup_f32_z(PTRUE32B, a); }
static inline vec_d vec_set1_d(const double a)    { return svdup_f64_z(PTRUE64B, a); }

static inline vec_s vec_bcast_s(float  const *a)  { return svdup_f32_z(PTRUE32B, a[0]); }
static inline vec_d vec_bcast_d(double const *a)  { return svdup_f64_z(PTRUE64B, a[0]); }

static inline vec_s vec_load_s (float  const *a)  { return svld1_f32(PTRUE32B, a);  }
static inline vec_d vec_load_d (double const *a)  { return svld1_f64(PTRUE64B, a);  }

static inline vec_s vec_loadu_s(float  const *a)  { return svld1_f32(PTRUE32B, a); }
static inline vec_d vec_loadu_d(double const *a)  { return svld1_f64(PTRUE64B, a); }

static inline void  vec_store_s (float  *a, const vec_s b)  { svst1_f32(PTRUE32B, a, b);  }
static inline void  vec_store_d (double *a, const vec_d b)  { svst1_f64(PTRUE64B, a, b);  }

static inline void  vec_storeu_s(float  *a, const vec_s b)  { svst1_f32(PTRUE32B, a, b); }
static inline void  vec_storeu_d(double *a, const vec_d b)  { svst1_f64(PTRUE64B, a, b); }

static inline vec_s vec_add_s(const vec_s a, const vec_s b) { return svadd_f32_z(PTRUE32B, a, b); }
static inline vec_d vec_add_d(const vec_d a, const vec_d b) { return svadd_f64_z(PTRUE64B, a, b); }

static inline vec_s vec_sub_s(const vec_s a, const vec_s b) { return svsub_f32_z(PTRUE32B, a, b); }
static inline vec_d vec_sub_d(const vec_d a, const vec_d b) { return svsub_f64_z(PTRUE64B, a, b); }

static inline vec_s vec_mul_s(const vec_s a, const vec_s b) { return svmul_f32_z(PTRUE32B, a, b); }
static inline vec_d vec_mul_d(const vec_d a, const vec_d b) { return svmul_f64_z(PTRUE64B, a, b); }

static inline vec_s vec_div_s(const vec_s a, const vec_s b) { return svdiv_f32_z(PTRUE32B, a, b); }
static inline vec_d vec_div_d(const vec_d a, const vec_d b) { return svdiv_f64_z(PTRUE64B, a, b); }

static inline vec_s vec_abs_s(const vec_s a) { return svabs_f32_z(PTRUE32B, a); }
static inline vec_d vec_abs_d(const vec_d a) { return svabs_f64_z(PTRUE64B, a); }

static inline vec_s vec_sqrt_s(const vec_s a) { return svsqrt_f32_z(PTRUE32B, a); }
static inline vec_d vec_sqrt_d(const vec_d a) { return svsqrt_f64_z(PTRUE64B, a); }

static inline vec_s vec_fmadd_s (const vec_s a, const vec_s b, const vec_s c) { return svmad_f32_z(PTRUE32B, a, b, c);  }
static inline vec_d vec_fmadd_d (const vec_d a, const vec_d b, const vec_d c) { return svmad_f64_z(PTRUE64B, a, b, c);  }

static inline vec_s vec_fnmadd_s(const vec_s a, const vec_s b, const vec_s c) { return svmsb_f32_z(PTRUE32B, a, b, c);  }
static inline vec_d vec_fnmadd_d(const vec_d a, const vec_d b, const vec_d c) { return svmsb_f64_z(PTRUE64B, a, b, c);  }

static inline vec_s vec_fmsub_s (const vec_s a, const vec_s b, const vec_s c) { return svnmsb_f32_z(PTRUE32B, a, b, c); }
static inline vec_d vec_fmsub_d (const vec_d a, const vec_d b, const vec_d c) { return svnmsb_f64_z(PTRUE64B, a, b, c); }

static inline vec_s vec_max_s(const vec_s a, const vec_s b) { return svmax_f32_z(PTRUE32B, a, b); }
static inline vec_d vec_max_d(const vec_d a, const vec_d b) { return svmax_f64_z(PTRUE64B, a, b); }

static inline vec_s vec_min_s(const vec_s a, const vec_s b) { return svmin_f32_z(PTRUE32B, a, b); }
static inline vec_d vec_min_d(const vec_d a, const vec_d b) { return svmin_f64_z(PTRUE64B, a, b); }

static inline vec_cmp_s vec_cmp_eq_s(const vec_s a, const vec_s b) { return svcmpeq_f32(PTRUE32B, a, b);  }
static inline vec_cmp_d vec_cmp_eq_d(const vec_d a, const vec_d b) { return svcmpeq_f64(PTRUE64B, a, b);  }

static inline vec_cmp_s vec_cmp_ne_s(const vec_s a, const vec_s b) { return svcmpne_f32(PTRUE32B, a, b); }
static inline vec_cmp_d vec_cmp_ne_d(const vec_d a, const vec_d b) { return svcmpne_f64(PTRUE64B, a, b); }

static inline vec_cmp_s vec_cmp_lt_s(const vec_s a, const vec_s b) { return svcmplt_f32(PTRUE32B, a, b); }
static inline vec_cmp_d vec_cmp_lt_d(const vec_d a, const vec_d b) { return svcmplt_f64(PTRUE64B, a, b); }

static inline vec_cmp_s vec_cmp_le_s(const vec_s a, const vec_s b) { return svcmple_f32(PTRUE32B, a, b); }
static inline vec_cmp_d vec_cmp_le_d(const vec_d a, const vec_d b) { return svcmple_f64(PTRUE64B, a, b); }

static inline vec_cmp_s vec_cmp_gt_s(const vec_s a, const vec_s b) { return svcmpgt_f32(PTRUE32B, a, b); }
static inline vec_cmp_d vec_cmp_gt_d(const vec_d a, const vec_d b) { return svcmpgt_f64(PTRUE64B, a, b); }

static inline vec_cmp_s vec_cmp_ge_s(const vec_s a, const vec_s b) { return svcmpge_f32(PTRUE32B, a, b); }
static inline vec_cmp_d vec_cmp_ge_d(const vec_d a, const vec_d b) { return svcmpge_f64(PTRUE64B, a, b); }

static inline vec_s vec_blend_s(const vec_s a, const vec_s b, const vec_cmp_s mask) { return svsel_f32(mask, b, a); }
static inline vec_d vec_blend_d(const vec_d a, const vec_d b, const vec_cmp_d mask) { return svsel_f64(mask, b, a); }

static inline float  vec_reduce_add_s(const vec_s a) { return svaddv_f32(PTRUE32B, a); }
static inline double vec_reduce_add_d(const vec_d a) { return svaddv_f64(PTRUE64B, a); }

static inline vec_s vec_arsqrt_s(const vec_s a)
{
    vec_s zero  = vec_zero_s();
    vec_s rsqrt = svrsqrte_f32(a);
    vec_cmp_s cmp0 = vec_cmp_eq_s(a, zero);
    return vec_blend_s(rsqrt, zero, cmp0);
}
static inline vec_d vec_arsqrt_d(const vec_d a)
{ 
    vec_d zero  = vec_zero_d();
    vec_d rsqrt = svrsqrte_f64(a);
    vec_cmp_d cmp0 = vec_cmp_eq_d(a, zero);
    return vec_blend_d(rsqrt, zero, cmp0);
}

#if !defined(SVE_LOOP_VEC_MATH)

vec_s _ZGVsNxv_logf(vec_s a);
vec_d _ZGVsNxv_log (vec_d a);
static inline vec_s vec_log_s  (vec_s a) { return _ZGVsNxv_logf(a);   }
static inline vec_d vec_log_d  (vec_d a) { return _ZGVsNxv_log (a);   }

vec_s _ZGVsNxv_log2f(vec_s a);
vec_d _ZGVsNxv_log2 (vec_d a);
static inline vec_s vec_log2_s (vec_s a) { return _ZGVsNxv_log2f(a);  }
static inline vec_d vec_log2_d (vec_d a) { return _ZGVsNxv_log2 (a);  }

vec_s _ZGVsNxv_log10f(vec_s a);
vec_d _ZGVsNxv_log10 (vec_d a);
static inline vec_s vec_log10_s(vec_s a) { return _ZGVsNxv_log10f(a); }
static inline vec_d vec_log10_d(vec_d a) { return _ZGVsNxv_log10 (a); }

vec_s _ZGVsNxv_expf(vec_s a);
vec_d _ZGVsNxv_exp (vec_d a);
static inline vec_s vec_exp_s  (vec_s a) { return _ZGVsNxv_expf(a);   }
static inline vec_d vec_exp_d  (vec_d a) { return _ZGVsNxv_exp (a);   }

vec_s _ZGVsNxv_exp2f(vec_s a);
vec_d _ZGVsNxv_exp2 (vec_d a);
static inline vec_s vec_exp2_s (vec_s a) { return _ZGVsNxv_exp2f(a);  }
static inline vec_d vec_exp2_d (vec_d a) { return _ZGVsNxv_exp2 (a);  }

vec_s _ZGVsNxv_exp10f(vec_s a);
vec_d _ZGVsNxv_exp10 (vec_d a);
static inline vec_s vec_exp10_s(vec_s a) { return _ZGVsNxv_exp10f(a); }
static inline vec_d vec_exp10_d(vec_d a) { return _ZGVsNxv_exp10 (a); }

vec_s _ZGVsNxv_powf(vec_s a, vec_s b);
vec_d _ZGVsNxv_pow (vec_d a, vec_d b);
static inline vec_s vec_pow_s  (vec_s a, vec_s b) { return _ZGVsNxv_powf(a, b); }
static inline vec_d vec_pow_d  (vec_d a, vec_d b) { return _ZGVsNxv_pow (a, b); }

vec_s _ZGVsNxv_sinf(vec_s a);
vec_d _ZGVsNxv_sin (vec_d a);
static inline vec_s vec_sin_s  (vec_s a) { return _ZGVsNxv_sinf(a);   }
static inline vec_d vec_sin_d  (vec_d a) { return _ZGVsNxv_sin (a);   }

vec_s _ZGVsNxv_cosf(vec_s a);
vec_d _ZGVsNxv_cos (vec_d a);
static inline vec_s vec_cos_s  (vec_s a) { return _ZGVsNxv_cosf(a);   }
static inline vec_d vec_cos_d  (vec_d a) { return _ZGVsNxv_cos (a);   }

vec_s _ZGVsNxv_erff(vec_s a);
vec_d _ZGVsNxv_erf (vec_d a);
static inline vec_s vec_erf_s  (vec_s a) { return _ZGVsNxv_erff(a);   }
static inline vec_d vec_erf_d  (vec_d a) { return _ZGVsNxv_erf (a);   }

#else  // Else of "#if !defined(SVE_LOOP_VEC_MATH)"

#warning sve_intrin_wrapper.h is using for-loop implementations for math functions.
static inline vec_s vec_log_s(vec_s a)
{
    int i;
    union vec_s_union u = {a}, res;
    for (i = 0; i < SIMD_LEN_S; i++) res.f[i] = logf(u.f[i]);
    return res.v;
}
static inline vec_d vec_log_d(vec_d a)
{
    int i;
    union vec_d_union u = {a}, res;
    for (i = 0; i < SIMD_LEN_D; i++) res.d[i] = log(u.d[i]);
    return res.v;
}

static inline vec_s vec_exp_s(vec_s a)
{
    int i;
    union vec_s_union u = {a}, res;
    for (i = 0; i < SIMD_LEN_S; i++) res.f[i] = expf(u.f[i]);
    return res.v;
}
static inline vec_d vec_exp_d(vec_d a)
{
    int i;
    union vec_d_union u = {a}, res;
    for (i = 0; i < SIMD_LEN_D; i++) res.d[i] = exp(u.d[i]);
    return res.v;
}

static inline vec_s vec_pow_s(vec_s a, vec_s b)
{
    int i;
    union vec_s_union ua = {a}, ub = {b}, res;
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

static inline vec_s vec_sin_s(vec_s a)
{
    int i;
    union vec_s_union ua = {a}, res;
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

static inline vec_s vec_cos_s(vec_s a)
{
    int i;
    union vec_s_union ua = {a}, res;
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

static inline vec_s vec_erf_s(vec_s a)
{
    int i;
    union vec_s_union ua = {a}, res;
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

static inline vec_s vec_log2_s (const vec_s a) { return vec_div_s(vec_log_s(a), vec_set1_s(M_LN2));  }
static inline vec_d vec_log2_d (const vec_d a) { return vec_div_d(vec_log_d(a), vec_set1_d(M_LN2));  }

static inline vec_s vec_log10_s(const vec_s a) { return vec_div_s(vec_log_s(a), vec_set1_s(M_LN10)); }
static inline vec_d vec_log10_d(const vec_d a) { return vec_div_d(vec_log_d(a), vec_set1_d(M_LN10)); }

static inline vec_s vec_exp2_s (const vec_s a) { return vec_exp_s(vec_mul_s(a, vec_set1_s(M_LN2)));  }
static inline vec_d vec_exp2_d (const vec_d a) { return vec_exp_d(vec_mul_d(a, vec_set1_d(M_LN2)));  }

static inline vec_s vec_exp10_s(const vec_s a) { return vec_exp_s(vec_mul_s(a, vec_set1_s(M_LN10))); }
static inline vec_d vec_exp10_d(const vec_d a) { return vec_exp_d(vec_mul_d(a, vec_set1_d(M_LN10))); }
#endif  // End of "#if !defined(SVE_LOOP_VEC_MATH)"

static inline vec_s vec_frsqrt_pf_s() { return vec_set1_s(1); }
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

static inline vec_s vec_frsqrt_s(const vec_s a)
{
    vec_s rsqrt = vec_arsqrt_s(a);
    vec_s rsqrt_work;
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
