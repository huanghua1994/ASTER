/* ================================================================================
Intrinsic function wrapper for Neon instruction set
References:
1. ARM Neon Intrinsics Reference : https://developer.arm.com/architectures/instruction-sets/simd-isas/neon/intrinsics
================================================================================ */ 

#ifndef __NEON_INTRIN_WRAPPER_H__
#define __NEON_INTRIN_WRAPPER_H__

#include <math.h>
#include <arm_neon.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef NEWTON_ITER
#define NEWTON_ITER 2   // Two Newton iterations is usually sufficient for rsqrt using double type
#endif

#if !defined(__ARM_NEON)
#error Your processor or compiler does not support NEON instruction set, cannot use this neon_intrin_wrapper.h
#endif

#define SIMD_LEN_S 4
#define SIMD_LEN_D 2
#define USE_NEON128

#define vec_s       float32x4_t
#define vec_d       float64x2_t
#define vec_cmp_s   uint32x4_t
#define vec_cmp_d   uint64x2_t

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

static inline vec_s vec_zero_s() { return vdupq_n_f32(0); }
static inline vec_d vec_zero_d() { return vdupq_n_f64(0); }

static inline vec_s vec_set1_s(const float  a)    { return vdupq_n_f32(a); }
static inline vec_d vec_set1_d(const double a)    { return vdupq_n_f64(a); }

static inline vec_s vec_bcast_s(float  const *a)  { return vdupq_n_f32(a[0]); }
static inline vec_d vec_bcast_d(double const *a)  { return vdupq_n_f64(a[0]); }

static inline vec_s vec_load_s (float  const *a)  { return vld1q_f32(a);  }
static inline vec_d vec_load_d (double const *a)  { return vld1q_f64(a);  }

static inline vec_s vec_loadu_s(float  const *a)  { return vld1q_f32(a); }
static inline vec_d vec_loadu_d(double const *a)  { return vld1q_f64(a); }

static inline void  vec_store_s (float  *a, const vec_s b)  { vst1q_f32(a, b);  }
static inline void  vec_store_d (double *a, const vec_d b)  { vst1q_f64(a, b);  }

static inline void  vec_storeu_s(float  *a, const vec_s b)  { vst1q_f32(a, b); }
static inline void  vec_storeu_d(double *a, const vec_d b)  { vst1q_f64(a, b); }

static inline vec_s vec_add_s(const vec_s a, const vec_s b) { return vaddq_f32(a, b); }
static inline vec_d vec_add_d(const vec_d a, const vec_d b) { return vaddq_f64(a, b); }

static inline vec_s vec_sub_s(const vec_s a, const vec_s b) { return vsubq_f32(a, b); }
static inline vec_d vec_sub_d(const vec_d a, const vec_d b) { return vsubq_f64(a, b); }

static inline vec_s vec_mul_s(const vec_s a, const vec_s b) { return vmulq_f32(a, b); }
static inline vec_d vec_mul_d(const vec_d a, const vec_d b) { return vmulq_f64(a, b); }

static inline vec_s vec_div_s(const vec_s a, const vec_s b) { return vdivq_f32(a, b); }
static inline vec_d vec_div_d(const vec_d a, const vec_d b) { return vdivq_f64(a, b); }

static inline vec_s vec_abs_s(const vec_s a) { return vabsq_f32(a); }
static inline vec_d vec_abs_d(const vec_d a) { return vabsq_f64(a); }

static inline vec_s vec_sqrt_s(const vec_s a) { return vsqrtq_f32(a); }
static inline vec_d vec_sqrt_d(const vec_d a) { return vsqrtq_f64(a); }

static inline vec_s vec_fmadd_s (const vec_s a, const vec_s b, const vec_s c) { return vfmaq_f32(c, a, b);  }
static inline vec_d vec_fmadd_d (const vec_d a, const vec_d b, const vec_d c) { return vfmaq_f64(c, a, b);  }

static inline vec_s vec_fnmadd_s(const vec_s a, const vec_s b, const vec_s c) { return vfmsq_f32(c, a, b);  }
static inline vec_d vec_fnmadd_d(const vec_d a, const vec_d b, const vec_d c) { return vfmsq_f64(c, a, b);  }

static inline vec_s vec_fmsub_s (const vec_s a, const vec_s b, const vec_s c) { return vnegq_f32(vfmsq_f32(c, a, b)); }
static inline vec_d vec_fmsub_d (const vec_d a, const vec_d b, const vec_d c) { return vnegq_f64(vfmsq_f64(c, a, b)); }

static inline vec_s vec_max_s(const vec_s a, const vec_s b) { return vmaxq_f32(a, b); }
static inline vec_d vec_max_d(const vec_d a, const vec_d b) { return vmaxq_f64(a, b); }

static inline vec_s vec_min_s(const vec_s a, const vec_s b) { return vminq_f32(a, b); }
static inline vec_d vec_min_d(const vec_d a, const vec_d b) { return vminq_f64(a, b); }

static inline vec_cmp_s vec_cmp_eq_s(const vec_s a, const vec_s b) { return vceqq_f32(a, b);  }
static inline vec_cmp_d vec_cmp_eq_d(const vec_d a, const vec_d b) { return vceqq_f64(a, b);  }

static inline vec_cmp_s vec_cmp_ne_s(const vec_s a, const vec_s b) { return vmvnq_u32(vceqq_f32(a, b)); }
static inline vec_cmp_d vec_cmp_ne_d(const vec_d a, const vec_d b) { return vandq_u64(vceqq_f64(a, b), vdupq_n_u64(0)); }

static inline vec_cmp_s vec_cmp_lt_s(const vec_s a, const vec_s b) { return vcltq_f32(a, b); }
static inline vec_cmp_d vec_cmp_lt_d(const vec_d a, const vec_d b) { return vcltq_f64(a, b); }

static inline vec_cmp_s vec_cmp_le_s(const vec_s a, const vec_s b) { return vcleq_f32(a, b); }
static inline vec_cmp_d vec_cmp_le_d(const vec_d a, const vec_d b) { return vcleq_f64(a, b); }

static inline vec_cmp_s vec_cmp_gt_s(const vec_s a, const vec_s b) { return vcgeq_f32(a, b); }
static inline vec_cmp_d vec_cmp_gt_d(const vec_d a, const vec_d b) { return vcgeq_f64(a, b); }

static inline vec_cmp_s vec_cmp_ge_s(const vec_s a, const vec_s b) { return vcgtq_f32(a, b); }
static inline vec_cmp_d vec_cmp_ge_d(const vec_d a, const vec_d b) { return vcgtq_f64(a, b); }

static inline vec_s vec_blend_s(const vec_s a, const vec_s b, const vec_cmp_s mask) { return vbslq_f32(mask, b, a); }
static inline vec_d vec_blend_d(const vec_d a, const vec_d b, const vec_cmp_d mask) { return vbslq_f64(mask, b, a); }

static inline float  vec_reduce_add_s(const vec_s a) { return vaddvq_f32(a); }
static inline double vec_reduce_add_d(const vec_d a) { return vaddvq_f64(a); }

static inline vec_s vec_arsqrt_s(const vec_s a)
{
    vec_s zero  = vec_zero_s();
    vec_s rsqrt = vrsqrteq_f32(a);
    vec_cmp_s cmp0 = vec_cmp_eq_s(a, zero);
    return vec_blend_s(rsqrt, zero, cmp0);
}
static inline vec_d vec_arsqrt_d(const vec_d a)
{ 
    vec_d zero  = vec_zero_d();
    vec_d rsqrt = vrsqrteq_f64(a);
    vec_cmp_d cmp0 = vec_cmp_eq_d(a, zero);
    return vec_blend_d(rsqrt, zero, cmp0);
}

#ifdef USE_SLEEF
#include "sleef.h"

static inline vec_s vec_log_s  (vec_s a) { return Sleef_logf4_u10advsimd(a);   }
static inline vec_d vec_log_d  (vec_d a) { return Sleef_logd2_u10advsimd(a);   }

static inline vec_s vec_log2_s (vec_s a) { return Sleef_log2f4_u10advsimd(a);  }
static inline vec_d vec_log2_d (vec_d a) { return Sleef_log2d2_u10advsimd(a);  }

static inline vec_s vec_log10_s(vec_s a) { return Sleef_log10f4_u10advsimd(a); }
static inline vec_d vec_log10_d(vec_d a) { return Sleef_log10d2_u10advsimd(a); }

static inline vec_s vec_exp_s  (vec_s a) { return Sleef_expf4_u10advsimd(a);   }
static inline vec_d vec_exp_d  (vec_d a) { return Sleef_expd2_u10advsimd(a);   }

static inline vec_s vec_exp2_s (vec_s a) { return Sleef_exp2f4_u10advsimd(a);  }
static inline vec_d vec_exp2_d (vec_d a) { return Sleef_exp2d2_u10advsimd(a);  }

static inline vec_s vec_exp10_s(vec_s a) { return Sleef_exp10f4_u10advsimd(a); }
static inline vec_d vec_exp10_d(vec_d a) { return Sleef_exp10d2_u10advsimd(a); }

static inline vec_s vec_pow_s  (vec_s a, vec_s b) { return Sleef_powf4_u10advsimd(a, b); }
static inline vec_d vec_pow_d  (vec_d a, vec_d b) { return Sleef_powd2_u10advsimd(a, b); }

static inline vec_s vec_sin_s  (vec_s a) { return Sleef_sinf4_u10advsimd(a);   }
static inline vec_d vec_sin_d  (vec_d a) { return Sleef_sind2_u10advsimd(a);   }

static inline vec_s vec_cos_s  (vec_s a) { return Sleef_cosf4_u10advsimd(a);   }
static inline vec_d vec_cos_d  (vec_d a) { return Sleef_cosd2_u10advsimd(a);   }

static inline vec_s vec_erf_s  (vec_s a) { return Sleef_erff4_u10advsimd(a);   }
static inline vec_d vec_erf_d  (vec_d a) { return Sleef_erfd2_u10advsimd(a);   }

#else  // Else of "#ifdef USE_SLEEF"

#warning SLEEF library not presented, neon_intrin_wrapper.h will use for-loop implementations.
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
#endif  // End of "#ifdef USE_SLEEF"

static inline vec_s vec_frsqrt_pf_s() { return vec_set1_s(1); }
static inline vec_d vec_frsqrt_pf_d() { return vec_set1_d(1); }

#define RSQRT_REFINE_F32(rsqrt_target, rsqrt_iter, rsqrt_work)  \
    do  \
    {   \
        rsqrt_work = vmulq_f32(rsqrt_target, rsqrt_iter);       \
        rsqrt_work = vrsqrtsq_f32(rsqrt_work, rsqrt_iter);      \
        rsqrt_iter = vmulq_f32(rsqrt_work, rsqrt_iter);         \
    } while (0)

#define RSQRT_REFINE_F64(rsqrt_target, rsqrt_iter, rsqrt_work)  \
    do  \
    {   \
        rsqrt_work = vmulq_f64(rsqrt_target, rsqrt_iter);       \
        rsqrt_work = vrsqrtsq_f64(rsqrt_work, rsqrt_iter);      \
        rsqrt_iter = vmulq_f64(rsqrt_work, rsqrt_iter);         \
    } while (0)

static inline vec_s vec_frsqrt_s(const vec_s a)
{
    vec_s rsqrt = vec_arsqrt_s(a);
    vec_s rsqrt_work;
    #if NEWTON_ITER >= 1
    RSQRT_REFINE_F32(a, rsqrt, rsqrt_work);
    #endif
    #if NEWTON_ITER >= 2
    RSQRT_REFINE_F32(a, rsqrt, rsqrt_work);
    #endif
    #if NEWTON_ITER >= 3
    RSQRT_REFINE_F32(a, rsqrt, rsqrt_work);
    #endif
    return rsqrt;
}
static inline vec_d vec_frsqrt_d(const vec_d a)
{
    vec_d rsqrt = vec_arsqrt_d(a);
    vec_d rsqrt_work;
    #if NEWTON_ITER >= 1
    RSQRT_REFINE_F64(a, rsqrt, rsqrt_work);
    #endif
    #if NEWTON_ITER >= 2
    RSQRT_REFINE_F64(a, rsqrt, rsqrt_work);
    #endif
    #if NEWTON_ITER >= 3
    RSQRT_REFINE_F64(a, rsqrt, rsqrt_work);
    #endif
    return rsqrt;
}

#ifdef __cplusplus
}
#endif

#endif
