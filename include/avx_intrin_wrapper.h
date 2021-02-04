/* ================================================================================
Intrinsic function wrapper for AVX, AVX-2, AVX-512 instruction sets
References:
1. Intel Intrinsic Guide    : https://software.intel.com/sites/landingpage/IntrinsicsGuide/
2. Compiler Explorer        : https://godbolt.org/
3. AVX vec_reduce_add_s     : https://stackoverflow.com/questions/13219146/how-to-sum-m256-horizontally
4. AVX vec_reduce_add_d     : https://www.oipapio.com/question-771803
5. Fast inverse square root : https://github.com/dmalhotra/pvfmm/blob/develop/include/intrin_wrapper.hpp
6. GCC SIMD math functions  : https://stackoverflow.com/questions/40475140/mathematical-functions-for-simd-registers
================================================================================ */ 

#ifndef __AVX_INTRIN_WRAPPER_H__
#define __AVX_INTRIN_WRAPPER_H__

#include <math.h>
#include <x86intrin.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef NEWTON_ITER
#define NEWTON_ITER 2   // Two Newton iterations is usually sufficient for rsqrt using double type
#endif

#if !defined(__AVX__)
#error Your processor or compiler does not support AVX instruction set, cannot use this avx_intrin_wrapper.h
#endif

#if !defined(USE_AVX) && !defined(USE_AVX512)
#if defined(__AVX512F__)
#define USE_AVX512
#elif defined(__AVX__)
#define USE_AVX
#endif
#endif

#if defined(__AVX512F__) && defined(USE_AVX)
#warning AVX-512 instruction set detected, but AVX/AVX2 instructions will be used due to -DUSE_AVX flag
#endif

#ifdef USE_AVX

#define SIMD_LEN_S  8
#define SIMD_LEN_D  4

#ifdef __AVX__

#define vec_s       __m256
#define vec_d       __m256d
#define vec_cmp_s   __m256
#define vec_cmp_d   __m256d

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

static inline vec_s vec_zero_s() { return _mm256_setzero_ps(); }
static inline vec_d vec_zero_d() { return _mm256_setzero_pd(); }

static inline vec_s vec_set1_s(const float  a)  { return _mm256_set1_ps(a); }
static inline vec_d vec_set1_d(const double a)  { return _mm256_set1_pd(a); }

static inline vec_s vec_bcast_s(float  const *a)  { return _mm256_broadcast_ss(a); }
static inline vec_d vec_bcast_d(double const *a)  { return _mm256_broadcast_sd(a); }

static inline vec_s vec_load_s (float  const *a)  { return _mm256_load_ps(a);  }
static inline vec_d vec_load_d (double const *a)  { return _mm256_load_pd(a);  }

static inline vec_s vec_loadu_s(float  const *a)  { return _mm256_loadu_ps(a); }
static inline vec_d vec_loadu_d(double const *a)  { return _mm256_loadu_pd(a); }

static inline void vec_store_s (float  *a, const vec_s b) { _mm256_store_ps(a, b);  }
static inline void vec_store_d (double *a, const vec_d b) { _mm256_store_pd(a, b);  }

static inline void vec_storeu_s(float  *a, const vec_s b) { _mm256_storeu_ps(a, b); }
static inline void vec_storeu_d(double *a, const vec_d b) { _mm256_storeu_pd(a, b); }

static inline vec_s vec_add_s(const vec_s a, const vec_s b) { return _mm256_add_ps(a, b); }
static inline vec_d vec_add_d(const vec_d a, const vec_d b) { return _mm256_add_pd(a, b); }

static inline vec_s vec_sub_s(const vec_s a, const vec_s b) { return _mm256_sub_ps(a, b); }
static inline vec_d vec_sub_d(const vec_d a, const vec_d b) { return _mm256_sub_pd(a, b); }

static inline vec_s vec_mul_s(const vec_s a, const vec_s b) { return _mm256_mul_ps(a, b); }
static inline vec_d vec_mul_d(const vec_d a, const vec_d b) { return _mm256_mul_pd(a, b); }

static inline vec_s vec_div_s(const vec_s a, const vec_s b) { return _mm256_div_ps(a, b); }
static inline vec_d vec_div_d(const vec_d a, const vec_d b) { return _mm256_div_pd(a, b); }

static inline vec_s vec_abs_s(const vec_s a) { return _mm256_max_ps(a, _mm256_sub_ps(_mm256_setzero_ps(), a)); }
static inline vec_d vec_abs_d(const vec_d a) { return _mm256_max_pd(a, _mm256_sub_pd(_mm256_setzero_pd(), a)); }

static inline vec_s vec_sqrt_s(const vec_s a) { return _mm256_sqrt_ps(a); }
static inline vec_d vec_sqrt_d(const vec_d a) { return _mm256_sqrt_pd(a); }

#ifdef __AVX2__
static inline vec_s vec_fmadd_s (const vec_s a, const vec_s b, const vec_s c) { return _mm256_fmadd_ps(a, b, c);  }
static inline vec_d vec_fmadd_d (const vec_d a, const vec_d b, const vec_d c) { return _mm256_fmadd_pd(a, b, c);  }

static inline vec_s vec_fnmadd_s(const vec_s a, const vec_s b, const vec_s c) { return _mm256_fnmadd_ps(a, b, c); }
static inline vec_d vec_fnmadd_d(const vec_d a, const vec_d b, const vec_d c) { return _mm256_fnmadd_pd(a, b, c); }

static inline vec_s vec_fmsub_s (const vec_s a, const vec_s b, const vec_s c) { return _mm256_fmsub_ps(a, b, c);  }
static inline vec_d vec_fmsub_d (const vec_d a, const vec_d b, const vec_d c) { return _mm256_fmsub_pd(a, b, c);  }
#else   // Else of "#ifdef __AVX2__"
static inline vec_s vec_fmadd_s (const vec_s a, const vec_s b, const vec_s c) { return _mm256_add_ps(_mm256_mul_ps(a, b), c); }
static inline vec_d vec_fmadd_d (const vec_d a, const vec_d b, const vec_d c) { return _mm256_add_pd(_mm256_mul_pd(a, b), c); }

static inline vec_s vec_fnmadd_s(const vec_s a, const vec_s b, const vec_s c) { return _mm256_sub_ps(c, _mm256_mul_ps(a, b)); }
static inline vec_d vec_fnmadd_d(const vec_d a, const vec_d b, const vec_d c) { return _mm256_sub_pd(c, _mm256_mul_pd(a, b)); }

static inline vec_s vec_fmsub_s (const vec_s a, const vec_s b, const vec_s c) { return _mm256_sub_ps(_mm256_mul_ps(a, b), c); }
static inline vec_d vec_fmsub_d (const vec_d a, const vec_d b, const vec_d c) { return _mm256_sub_pd(_mm256_mul_pd(a, b), c); }
#endif  // End of "#ifdef __AVX2__"

static inline vec_s vec_max_s(const vec_s a, const vec_s b) { return _mm256_max_ps(a, b); }
static inline vec_d vec_max_d(const vec_d a, const vec_d b) { return _mm256_max_pd(a, b); }

static inline vec_s vec_min_s(const vec_s a, const vec_s b) { return _mm256_min_ps(a, b); }
static inline vec_d vec_min_d(const vec_d a, const vec_d b) { return _mm256_min_pd(a, b); }

static inline vec_s vec_cmp_eq_s(const vec_s a, const vec_s b) { return _mm256_cmp_ps(a, b, _CMP_EQ_OS);  }
static inline vec_d vec_cmp_eq_d(const vec_d a, const vec_d b) { return _mm256_cmp_pd(a, b, _CMP_EQ_OS);  }

static inline vec_s vec_cmp_ne_s(const vec_s a, const vec_s b) { return _mm256_cmp_ps(a, b, _CMP_NEQ_OS); }
static inline vec_d vec_cmp_ne_d(const vec_d a, const vec_d b) { return _mm256_cmp_pd(a, b, _CMP_NEQ_OS); }

static inline vec_s vec_cmp_lt_s(const vec_s a, const vec_s b) { return _mm256_cmp_ps(a, b, _CMP_LT_OS);  }
static inline vec_d vec_cmp_lt_d(const vec_d a, const vec_d b) { return _mm256_cmp_pd(a, b, _CMP_LT_OS);  }

static inline vec_s vec_cmp_le_s(const vec_s a, const vec_s b) { return _mm256_cmp_ps(a, b, _CMP_LE_OS);  }
static inline vec_d vec_cmp_le_d(const vec_d a, const vec_d b) { return _mm256_cmp_pd(a, b, _CMP_LE_OS);  }

static inline vec_s vec_cmp_gt_s(const vec_s a, const vec_s b) { return _mm256_cmp_ps(a, b, _CMP_GT_OS);  }
static inline vec_d vec_cmp_gt_d(const vec_d a, const vec_d b) { return _mm256_cmp_pd(a, b, _CMP_GT_OS);  }

static inline vec_s vec_cmp_ge_s(const vec_s a, const vec_s b) { return _mm256_cmp_ps(a, b, _CMP_GE_OS);  }
static inline vec_d vec_cmp_ge_d(const vec_d a, const vec_d b) { return _mm256_cmp_pd(a, b, _CMP_GE_OS);  }

static inline vec_s vec_blend_s(const vec_s a, const vec_s b, const vec_s mask) { return _mm256_blendv_ps(a, b, mask); }
static inline vec_d vec_blend_d(const vec_d a, const vec_d b, const vec_d mask) { return _mm256_blendv_pd(a, b, mask); }

static inline float vec_reduce_add_s(const vec_s a) 
{
    __m128 hi4  = _mm256_extractf128_ps(a, 1);
    __m128 lo4  = _mm256_castps256_ps128(a);
    __m128 sum4 = _mm_add_ps(lo4, hi4);
    __m128 lo2  = sum4;
    __m128 hi2  = _mm_movehl_ps(sum4, sum4);
    __m128 sum2 = _mm_add_ps(lo2, hi2);
    __m128 lo   = sum2;
    __m128 hi   = _mm_shuffle_ps(sum2, sum2, 0x1);
    __m128 sum  = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}
static inline double vec_reduce_add_d(const vec_d a) 
{
    __m128d lo = _mm256_castpd256_pd128(a);
    __m128d hi = _mm256_extractf128_pd(a, 1); 
    lo = _mm_add_pd(lo, hi);
    __m128d hi64 = _mm_unpackhi_pd(lo, lo);
    return  _mm_cvtsd_f64(_mm_add_sd(lo, hi64));
}

static inline vec_s vec_arsqrt_s(const vec_s a)
{
    vec_s rsqrt = _mm256_rsqrt_ps(a);
    vec_s cmp0  = _mm256_cmp_ps(a, vec_zero_s(), _CMP_EQ_OS);
    return _mm256_andnot_ps(cmp0, rsqrt);
}
static inline vec_d vec_arsqrt_d(const vec_d a)
{ 
    __m128 a_s     = _mm256_cvtpd_ps(a);
    __m128 rsqrt_s = _mm_rsqrt_ps(a_s);
    __m128 cmp0    = _mm_cmpeq_ps(a_s, _mm_setzero_ps());
    __m128 ret_s   = _mm_andnot_ps(cmp0, rsqrt_s);
    return _mm256_cvtps_pd(ret_s); 
}

#ifdef __INTEL_COMPILER
static inline vec_s vec_log_s  (const vec_s a) { return _mm256_log_ps(a);   }
static inline vec_d vec_log_d  (const vec_d a) { return _mm256_log_pd(a);   }

static inline vec_s vec_log2_s (const vec_s a) { return _mm256_log2_ps(a);  }
static inline vec_d vec_log2_d (const vec_d a) { return _mm256_log2_pd(a);  }

static inline vec_s vec_log10_s(const vec_s a) { return _mm256_log10_ps(a); }
static inline vec_d vec_log10_d(const vec_d a) { return _mm256_log10_pd(a); }

static inline vec_s vec_exp_s  (const vec_s a) { return _mm256_exp_ps(a);   }
static inline vec_d vec_exp_d  (const vec_d a) { return _mm256_exp_pd(a);   }

static inline vec_s vec_exp2_s (const vec_s a) { return _mm256_exp2_ps(a);  }
static inline vec_d vec_exp2_d (const vec_d a) { return _mm256_exp2_pd(a);  }

static inline vec_s vec_exp10_s(const vec_s a) { return _mm256_exp10_ps(a); }
static inline vec_d vec_exp10_d(const vec_d a) { return _mm256_exp10_pd(a); }

static inline vec_s vec_pow_s  (const vec_s a, const vec_s b) { return _mm256_pow_ps(a, b); }
static inline vec_d vec_pow_d  (const vec_d a, const vec_d b) { return _mm256_pow_pd(a, b); }

static inline vec_s vec_sin_s  (const vec_s a) { return _mm256_sin_ps(a);   }
static inline vec_d vec_sin_d  (const vec_d a) { return _mm256_sin_pd(a);   }

static inline vec_s vec_cos_s  (const vec_s a) { return _mm256_cos_ps(a);   }
static inline vec_d vec_cos_d  (const vec_d a) { return _mm256_cos_pd(a);   }

static inline vec_s vec_erf_s  (const vec_s a) { return _mm256_erf_ps(a);   }
static inline vec_d vec_erf_d  (const vec_d a) { return _mm256_erf_pd(a);   }

#else  // Else of "#ifdef __INTEL_COMPILER"

#if __GLIBC__ >= 2 && __GLIBC_MINOR__ >= 22
vec_s _ZGVdN8v_logf(vec_s a);
vec_d _ZGVdN4v_log (vec_d a);
static inline vec_s vec_log_s  (const vec_s a) { return _ZGVdN8v_logf(a);   }
static inline vec_d vec_log_d  (const vec_d a) { return _ZGVdN4v_log (a);   }

vec_s _ZGVdN8v_expf(vec_s a);
vec_d _ZGVdN4v_exp (vec_d a);
static inline vec_s vec_exp_s  (const vec_s a) { return _ZGVdN8v_expf(a);   }
static inline vec_d vec_exp_d  (const vec_d a) { return _ZGVdN4v_exp (a);   }

vec_s _ZGVdN8vv_powf(vec_s a, vec_s b);
vec_d _ZGVdN4vv_pow (vec_d a, vec_d b);
static inline vec_s vec_pow_s  (const vec_s a, const vec_s b) { return _ZGVdN8vv_powf(a, b); }
static inline vec_d vec_pow_d  (const vec_d a, const vec_d b) { return _ZGVdN4vv_pow (a, b); }

vec_s _ZGVdN8v_sinf(vec_s a);
vec_d _ZGVdN4v_sin (vec_d a);
static inline vec_s vec_sin_s  (const vec_s a) { return _ZGVdN8v_sinf(a);   }
static inline vec_d vec_sin_d  (const vec_d a) { return _ZGVdN4v_sin (a);   }

vec_s _ZGVdN8v_cosf(vec_s a);
vec_d _ZGVdN4v_cos (vec_d a);
static inline vec_s vec_cos_s  (const vec_s a) { return _ZGVdN8v_cosf(a);   }
static inline vec_d vec_cos_d  (const vec_d a) { return _ZGVdN4v_cos (a);   }

#else   // Else of "#if __GLIBC__ >= 2 && __GLIBC_MINOR__ >= 22"

#warning Your compiler or GLIBC does not support vectorized log(), pow(), and exp(), avx_intrin_wrapper.h will use for-loop implementations.
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

#endif  // End of "#if __GLIBC__ >= 2 && __GLIBC_MINOR__ >= 22"

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

#endif  // End of "#ifdef __INTEL_COMPILER"

#endif  // End of "#ifdef __AVX__"
#endif  // End of "#ifdef USE_AVX"


#ifdef USE_AVX512

#define SIMD_LEN_S  16
#define SIMD_LEN_D  8

#ifdef __AVX512F__

#define vec_s       __m512
#define vec_d       __m512d
#define vec_cmp_s   __mmask16
#define vec_cmp_d   __mmask8

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

static inline vec_s vec_zero_s() { return _mm512_setzero_ps(); }
static inline vec_d vec_zero_d() { return _mm512_setzero_pd(); }

static inline vec_s vec_set1_s(const float  a)  { return _mm512_set1_ps(a); }
static inline vec_d vec_set1_d(const double a)  { return _mm512_set1_pd(a); }

static inline vec_s vec_bcast_s(float  const *a)  { return _mm512_set1_ps(a[0]); }
static inline vec_d vec_bcast_d(double const *a)  { return _mm512_set1_pd(a[0]); }

static inline vec_s vec_load_s (float  const *a)  { return _mm512_load_ps(a);  }
static inline vec_d vec_load_d (double const *a)  { return _mm512_load_pd(a);  }

static inline vec_s vec_loadu_s(float  const *a)  { return _mm512_loadu_ps(a); }
static inline vec_d vec_loadu_d(double const *a)  { return _mm512_loadu_pd(a); }

static inline void  vec_store_s (float  *a, const vec_s b) { _mm512_store_ps(a, b);  }
static inline void  vec_store_d (double *a, const vec_d b) { _mm512_store_pd(a, b);  }

static inline void  vec_storeu_s(float  *a, const vec_s b) { _mm512_storeu_ps(a, b); }
static inline void  vec_storeu_d(double *a, const vec_d b) { _mm512_storeu_pd(a, b); }

static inline vec_s vec_add_s(const vec_s a, const vec_s b) { return _mm512_add_ps(a, b); }
static inline vec_d vec_add_d(const vec_d a, const vec_d b) { return _mm512_add_pd(a, b); }

static inline vec_s vec_sub_s(const vec_s a, const vec_s b) { return _mm512_sub_ps(a, b); }
static inline vec_d vec_sub_d(const vec_d a, const vec_d b) { return _mm512_sub_pd(a, b); }

static inline vec_s vec_mul_s(const vec_s a, const vec_s b) { return _mm512_mul_ps(a, b); }
static inline vec_d vec_mul_d(const vec_d a, const vec_d b) { return _mm512_mul_pd(a, b); }

static inline vec_s vec_div_s(const vec_s a, const vec_s b) { return _mm512_div_ps(a, b); }
static inline vec_d vec_div_d(const vec_d a, const vec_d b) { return _mm512_div_pd(a, b); }

static inline vec_s vec_abs_s(const vec_s a) { return _mm512_abs_ps(a); }
static inline vec_d vec_abs_d(const vec_d a) { return _mm512_abs_pd(a); }

static inline vec_s vec_sqrt_s(const vec_s a) { return _mm512_sqrt_ps(a); }
static inline vec_d vec_sqrt_d(const vec_d a) { return _mm512_sqrt_pd(a); }

static inline vec_s vec_fmadd_s (const vec_s a, const vec_s b, const vec_s c) { return _mm512_fmadd_ps(a, b, c);  }
static inline vec_d vec_fmadd_d (const vec_d a, const vec_d b, const vec_d c) { return _mm512_fmadd_pd(a, b, c);  }

static inline vec_s vec_fnmadd_s(const vec_s a, const vec_s b, const vec_s c) { return _mm512_fnmadd_ps(a, b, c); }
static inline vec_d vec_fnmadd_d(const vec_d a, const vec_d b, const vec_d c) { return _mm512_fnmadd_pd(a, b, c); }

static inline vec_s vec_fmsub_s (const vec_s a, const vec_s b, const vec_s c) { return _mm512_fmsub_ps(a, b, c);  }
static inline vec_d vec_fmsub_d (const vec_d a, const vec_d b, const vec_d c) { return _mm512_fmsub_pd(a, b, c);  }

static inline vec_s vec_max_s(const vec_s a, const vec_s b) { return _mm512_max_ps(a, b); }
static inline vec_d vec_max_d(const vec_d a, const vec_d b) { return _mm512_max_pd(a, b); }

static inline vec_s vec_min_s(const vec_s a, const vec_s b) { return _mm512_min_ps(a, b); }
static inline vec_d vec_min_d(const vec_d a, const vec_d b) { return _mm512_min_pd(a, b); }

static inline vec_cmp_s vec_cmp_eq_s(const vec_s a, const vec_s b) { return _mm512_cmp_ps_mask(a, b, _CMP_EQ_OS);  }
static inline vec_cmp_d vec_cmp_eq_d(const vec_d a, const vec_d b) { return _mm512_cmp_pd_mask(a, b, _CMP_EQ_OS);  }

static inline vec_cmp_s vec_cmp_ne_s(const vec_s a, const vec_s b) { return _mm512_cmp_ps_mask(a, b, _CMP_NEQ_OS); }
static inline vec_cmp_d vec_cmp_ne_d(const vec_d a, const vec_d b) { return _mm512_cmp_pd_mask(a, b, _CMP_NEQ_OS); }

static inline vec_cmp_s vec_cmp_lt_s(const vec_s a, const vec_s b) { return _mm512_cmp_ps_mask(a, b, _CMP_LT_OS);  }
static inline vec_cmp_d vec_cmp_lt_d(const vec_d a, const vec_d b) { return _mm512_cmp_pd_mask(a, b, _CMP_LT_OS);  }

static inline vec_cmp_s vec_cmp_le_s(const vec_s a, const vec_s b) { return _mm512_cmp_ps_mask(a, b, _CMP_LE_OS);  }
static inline vec_cmp_d vec_cmp_le_d(const vec_d a, const vec_d b) { return _mm512_cmp_pd_mask(a, b, _CMP_LE_OS);  }

static inline vec_cmp_s vec_cmp_gt_s(const vec_s a, const vec_s b) { return _mm512_cmp_ps_mask(a, b, _CMP_GT_OS);  }
static inline vec_cmp_d vec_cmp_gt_d(const vec_d a, const vec_d b) { return _mm512_cmp_pd_mask(a, b, _CMP_GT_OS);  }

static inline vec_cmp_s vec_cmp_ge_s(const vec_s a, const vec_s b) { return _mm512_cmp_ps_mask(a, b, _CMP_GE_OS);  }
static inline vec_cmp_d vec_cmp_ge_d(const vec_d a, const vec_d b) { return _mm512_cmp_pd_mask(a, b, _CMP_GE_OS);  }

static inline vec_s vec_blend_s(const vec_s a, const vec_s b, const vec_cmp_s mask) { return _mm512_mask_blend_ps(mask, a, b); }
static inline vec_d vec_blend_d(const vec_d a, const vec_d b, const vec_cmp_d mask) { return _mm512_mask_blend_pd(mask, a, b); }

static inline float  vec_reduce_add_s(const vec_s a) { return _mm512_reduce_add_ps(a); }
static inline double vec_reduce_add_d(const vec_d a) { return _mm512_reduce_add_pd(a); }

#ifdef __AVX512ER__
static inline vec_s vec_arsqrt_s(const vec_s a)
{
    vec_s zero  = vec_zero_s();
    vec_s rsqrt = _mm512_rsqrt28_ps(a);
    vec_cmp_s cmp0 = _mm512_cmp_ps_mask(a, zero, _CMP_EQ_OS);
    return _mm512_mask_mov_ps(rsqrt, cmp0, zero);
}
static inline vec_d vec_arsqrt_d(const vec_d a)
{ 
    vec_d zero  = vec_zero_d();
    vec_d rsqrt = _mm512_rsqrt28_pd(a);
    vec_cmp_d cmp0 = _mm512_cmp_pd_mask(a, zero, _CMP_EQ_OS);
    return _mm512_mask_mov_pd(rsqrt, cmp0, zero);
}
#else   // Else of "#ifdef __AVX512ER__"
static inline vec_s vec_arsqrt_s(const vec_s a)
{
    vec_s zero  = vec_zero_s();
    vec_s rsqrt = _mm512_rsqrt14_ps(a);
    vec_cmp_s cmp0 = _mm512_cmp_ps_mask(a, zero, _CMP_EQ_OS);
    return _mm512_mask_mov_ps(rsqrt, cmp0, zero);
}
static inline vec_d vec_arsqrt_d(const vec_d a)
{ 
    vec_d zero  = vec_zero_d();
    vec_d rsqrt = _mm512_rsqrt14_pd(a);
    vec_cmp_d cmp0 = _mm512_cmp_pd_mask(a, zero, _CMP_EQ_OS);
    return _mm512_mask_mov_pd(rsqrt, cmp0, zero);
}
#endif  // End of "#ifdef __AVX512ER__"

#ifdef __INTEL_COMPILER
static inline vec_s vec_log_s  (const vec_s a) { return _mm512_log_ps(a);   }
static inline vec_d vec_log_d  (const vec_d a) { return _mm512_log_pd(a);   }

static inline vec_s vec_log2_s (const vec_s a) { return _mm512_log2_ps(a);  }
static inline vec_d vec_log2_d (const vec_d a) { return _mm512_log2_pd(a);  }

static inline vec_s vec_log10_s(const vec_s a) { return _mm512_log10_ps(a); }
static inline vec_d vec_log10_d(const vec_d a) { return _mm512_log10_pd(a); }

static inline vec_s vec_exp_s  (const vec_s a) { return _mm512_exp_ps(a);   }
static inline vec_d vec_exp_d  (const vec_d a) { return _mm512_exp_pd(a);   }

static inline vec_s vec_exp2_s (const vec_s a) { return _mm512_exp2_ps(a);  }
static inline vec_d vec_exp2_d (const vec_d a) { return _mm512_exp2_pd(a);  }

static inline vec_s vec_exp10_s(const vec_s a) { return _mm512_exp10_ps(a); }
static inline vec_d vec_exp10_d(const vec_d a) { return _mm512_exp10_pd(a); }

static inline vec_s vec_pow_s  (const vec_s a, const vec_s b) { return _mm512_pow_ps(a, b); }
static inline vec_d vec_pow_d  (const vec_d a, const vec_d b) { return _mm512_pow_pd(a, b); }

static inline vec_s vec_sin_s  (const vec_s a) { return _mm512_sin_ps(a);   }
static inline vec_d vec_sin_d  (const vec_d a) { return _mm512_sin_pd(a);   }

static inline vec_s vec_cos_s  (const vec_s a) { return _mm512_cos_ps(a);   }
static inline vec_d vec_cos_d  (const vec_d a) { return _mm512_cos_pd(a);   }

static inline vec_s vec_erf_s  (const vec_s a) { return _mm512_erf_ps(a);   }
static inline vec_d vec_erf_d  (const vec_d a) { return _mm512_erf_pd(a);   }

#else   // Else of "#ifdef __INTEL_COMPILER"

#if __GLIBC__ >= 2 && __GLIBC_MINOR__ >= 22
vec_s _ZGVeN16v_logf(vec_s a);
vec_d _ZGVeN8v_log  (vec_d a);
static inline vec_s vec_log_s  (const vec_s a) { return _ZGVeN16v_logf(a);  }
static inline vec_d vec_log_d  (const vec_d a) { return _ZGVeN8v_log  (a);  }

vec_s _ZGVeN16v_expf(vec_s a);
vec_d _ZGVeN8v_exp  (vec_d a);
static inline vec_s vec_exp_s  (const vec_s a) { return _ZGVeN16v_expf(a);  }
static inline vec_d vec_exp_d  (const vec_d a) { return _ZGVeN8v_exp  (a);  }

vec_s _ZGVeN16vv_powf(vec_s a, vec_s b);
vec_d _ZGVeN8vv_pow  (vec_d a, vec_d b);
static inline vec_s vec_pow_s  (const vec_s a, const vec_s b) { return _ZGVeN16vv_powf(a, b); }
static inline vec_d vec_pow_d  (const vec_d a, const vec_d b) { return _ZGVeN8vv_pow  (a, b); }

vec_s _ZGVdN16v_sinf(vec_s a);
vec_d _ZGVdN8v_sin  (vec_d a);
static inline vec_s vec_sin_s  (const vec_s a) { return _ZGVdN16v_sinf(a);  }
static inline vec_d vec_sin_d  (const vec_d a) { return _ZGVdN8v_sin  (a);  }

vec_s _ZGVdN16v_cosf(vec_s a);
vec_d _ZGVdN8v_cos  (vec_d a);
static inline vec_s vec_cos_s  (const vec_s a) { return _ZGVdN16v_cosf(a);  }
static inline vec_d vec_cos_d  (const vec_d a) { return _ZGVdN8v_cos (a);   }

#else   // Else of "#if __GLIBC__ >= 2 && __GLIBC_MINOR__ >= 22"

#warning Your compiler or GLIBC does not support vectorized log(), pow(), and exp(), avx_intrin_wrapper.h will use for-loop implementations.
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
#endif  // End of "#if __GLIBC__ >= 2 && __GLIBC_MINOR__ >= 22"

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

#endif  // End of "#ifdef __INTEL_COMPILER"

#endif  // End of "#ifdef __AVX512F__"
#endif  // End of "#ifdef USE_AVX512"


#ifndef __INTEL_COMPILER
static inline vec_s vec_log2_s (const vec_s a) { return vec_div_s(vec_log_s(a), vec_set1_s(M_LN2));  }
static inline vec_d vec_log2_d (const vec_d a) { return vec_div_d(vec_log_d(a), vec_set1_d(M_LN2));  }

static inline vec_s vec_log10_s(const vec_s a) { return vec_div_s(vec_log_s(a), vec_set1_s(M_LN10)); }
static inline vec_d vec_log10_d(const vec_d a) { return vec_div_d(vec_log_d(a), vec_set1_d(M_LN10)); }

static inline vec_s vec_exp2_s (const vec_s a) { return vec_exp_s(vec_mul_s(a, vec_set1_s(M_LN2)));  }
static inline vec_d vec_exp2_d (const vec_d a) { return vec_exp_d(vec_mul_d(a, vec_set1_d(M_LN2)));  }

static inline vec_s vec_exp10_s(const vec_s a) { return vec_exp_s(vec_mul_s(a, vec_set1_s(M_LN10))); }
static inline vec_d vec_exp10_d(const vec_d a) { return vec_exp_d(vec_mul_d(a, vec_set1_d(M_LN10))); }
#endif  // End of #ifdef __INTEL_COMPILER

// Newton iteration step for reverse square root, rsqrt' = 0.5 * rsqrt * (C - a * rsqrt^2),
// 0.5 is ignored here and need to be adjusted outside. 
static inline vec_s vec_rsqrt_ntit_s(const vec_s a, vec_s rsqrt, const float  C_)
{
    vec_s C  = vec_set1_s(C_);
    vec_s t1 = vec_mul_s(rsqrt, rsqrt);
    vec_s t2 = vec_fnmadd_s(a, t1, C);
    return vec_mul_s(rsqrt, t2);
}
static inline vec_d vec_rsqrt_ntit_d(const vec_d a, vec_d rsqrt, const double C_)
{
    vec_d C  = vec_set1_d(C_);
    vec_d t1 = vec_mul_d(rsqrt, rsqrt);
    vec_d t2 = vec_fnmadd_d(a, t1, C);
    return vec_mul_d(rsqrt, t2);
}

static inline vec_s vec_frsqrt_pf_s()
{
    float newton_pf = 1.0;
    for (int i = 0; i < NEWTON_ITER; i++)
        newton_pf = 2.0 * newton_pf * newton_pf * newton_pf;
    newton_pf = 1.0 / newton_pf;
    return vec_set1_s(newton_pf);
}
static inline vec_d vec_frsqrt_pf_d()
{
    double newton_pf = 1.0;
    for (int i = 0; i < NEWTON_ITER; i++)
        newton_pf = 2.0 * newton_pf * newton_pf * newton_pf;
    newton_pf = 1.0 / newton_pf;
    return vec_set1_d(newton_pf);
}

static inline vec_s vec_frsqrt_s(const vec_s a)
{
    vec_s rsqrt = vec_arsqrt_s(a);
    #if NEWTON_ITER >= 1
    rsqrt = vec_rsqrt_ntit_s(a, rsqrt, 3);
    #endif
    #if NEWTON_ITER >= 2
    rsqrt = vec_rsqrt_ntit_s(a, rsqrt, 12);
    #endif
    #if NEWTON_ITER >= 3
    rsqrt = vec_rsqrt_ntit_s(a, rsqrt, 768);
    #endif
    return rsqrt;
}
static inline vec_d vec_frsqrt_d(const vec_d a)
{
    vec_d rsqrt = vec_arsqrt_d(a);
    #if NEWTON_ITER >= 1
    rsqrt = vec_rsqrt_ntit_d(a, rsqrt, 3);
    #endif
    #if NEWTON_ITER >= 2
    rsqrt = vec_rsqrt_ntit_d(a, rsqrt, 12);
    #endif
    #if NEWTON_ITER >= 3
    rsqrt = vec_rsqrt_ntit_d(a, rsqrt, 768);
    #endif
    return rsqrt;
}


#ifdef __cplusplus
}
#endif

#endif  // End of header file 
