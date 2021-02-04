#ifndef __DTYPE_WRAPPER_H__
#define __DTYPE_WRAPPER_H__

#if !defined(ASTER_DTYPE_DOUBLE) && !defined(ASTER_DTYPE_FLOAT)
#warning Neither ASTER_DTYPE_DOUBLE nor ASTER_DTYPE_FLOAT is defined. ASTER will use double for *_t
#define ASTER_DTYPE_DOUBLE
#endif

#ifdef ASTER_DTYPE_DOUBLE
#define SIMD_LEN_T          SIMD_LEN_D
#define vec_t               vec_d
#define vec_cmp_t           vec_cmp_d
#define vec_zero_t          vec_zero_d
#define vec_set1_t          vec_set1_d
#define vec_bcast_t         vec_bcast_d
#define vec_load_t          vec_load_d
#define vec_loadu_t         vec_loadu_d
#define vec_store_t         vec_store_d
#define vec_storeu_t        vec_storeu_d
#define vec_add_t           vec_add_d
#define vec_sub_t           vec_sub_d
#define vec_mul_t           vec_mul_d
#define vec_div_t           vec_div_d
#define vec_abs_t           vec_abs_d
#define vec_abs_t           vec_abs_d
#define vec_sqrt_t          vec_sqrt_d
#define vec_fmadd_t         vec_fmadd_d
#define vec_fmsub_t         vec_fmsub_d
#define vec_fnmadd_t        vec_fnmadd_d
#define vec_max_t           vec_max_d
#define vec_min_t           vec_min_d
#define vec_cmp_eq_t        vec_cmp_eq_d
#define vec_cmp_ne_t        vec_cmp_ne_d
#define vec_cmp_lt_t        vec_cmp_lt_d
#define vec_cmp_le_t        vec_cmp_le_d
#define vec_cmp_gt_t        vec_cmp_gt_d
#define vec_cmp_ge_t        vec_cmp_ge_d
#define vec_blend_t         vec_blend_d
#define vec_reduce_add_t    vec_reduce_add_d
#define vec_arsqrt_t        vec_arsqrt_d
#define vec_log_t           vec_log_d
#define vec_log2_t          vec_log2_d
#define vec_log10_t         vec_log10_d
#define vec_exp_t           vec_exp_d
#define vec_exp2_t          vec_exp2_d
#define vec_exp10_t         vec_exp10_d
#define vec_pow_t           vec_pow_d
#define vec_sin_t           vec_sin_d
#define vec_cos_t           vec_cos_d
#define vec_erf_t           vec_erf_d
#define vec_frsqrt_pf_t     vec_frsqrt_pf_d
#define vec_frsqrt_t        vec_frsqrt_d
#endif  // "#ifdef ASTER_DTYPE_DOUBLE"

#ifdef ASTER_DTYPE_FLOAT
#define SIMD_LEN_T          SIMD_LEN_S
#define vec_t               vec_s
#define vec_cmp_t           vec_cmp_s
#define vec_zero_t          vec_zero_s
#define vec_set1_t          vec_set1_s
#define vec_bcast_t         vec_bcast_s
#define vec_load_t          vec_load_s
#define vec_loadu_t         vec_loadu_s
#define vec_store_t         vec_store_s
#define vec_storeu_t        vec_storeu_s
#define vec_add_t           vec_add_s
#define vec_sub_t           vec_sub_s
#define vec_mul_t           vec_mul_s
#define vec_div_t           vec_div_s
#define vec_abs_t           vec_abs_s
#define vec_abs_t           vec_abs_s
#define vec_sqrt_t          vec_sqrt_s
#define vec_fmadd_t         vec_fmadd_s
#define vec_fmsub_t         vec_fmsub_s
#define vec_fnmadd_t        vec_fnmadd_s
#define vec_max_t           vec_max_s
#define vec_min_t           vec_min_s
#define vec_cmp_eq_t        vec_cmp_eq_s
#define vec_cmp_ne_t        vec_cmp_ne_s
#define vec_cmp_lt_t        vec_cmp_lt_s
#define vec_cmp_le_t        vec_cmp_le_s
#define vec_cmp_gt_t        vec_cmp_gt_s
#define vec_cmp_ge_t        vec_cmp_ge_s
#define vec_blend_t         vec_blend_s
#define vec_reduce_add_t    vec_reduce_add_s
#define vec_arsqrt_t        vec_arsqrt_s
#define vec_log_t           vec_log_s
#define vec_log2_t          vec_log2_s
#define vec_log10_t         vec_log10_s
#define vec_exp_t           vec_exp_s
#define vec_exp2_t          vec_exp2_s
#define vec_exp10_t         vec_exp10_s
#define vec_pow_t           vec_pow_s
#define vec_sin_t           vec_sin_s
#define vec_cos_t           vec_cos_s
#define vec_erf_t           vec_erf_s
#define vec_frsqrt_pf_t     vec_frsqrt_pf_s
#define vec_frsqrt_t        vec_frsqrt_s
#endif  // "#ifdef ASTER_DTYPE_FLOAT"


#endif
