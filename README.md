# ASTER
Acceleration with Simd inTrinsic, Easy and Reusable



ASTER is a C-header only library that provides a set APIs for frequently used SIMD arithmetical operations. Currently ASTER supports the following instruction sets: AVX, AVX2, AVX-512, and SVE.



Vector types:

* `vec_s`: data vector for single-precision floating point numbers (float, fp32), each vector has `SIMD_LEN_S` lanes (elements)
* `vec_d`: data vector for double-precision floating point numbers (double, fp64), each vector has `SIMD_LEN_D` lanes (elements)
* `vec_t`: if `ASTER_DTYPE_DOUBLE` / `ASTER_DTYPE_FLOAT` is defined before including `aster.h`, `vec_t` == `vec_d` / `vec_s`
* `vec_cmp_s`: comparing & mask vector for `vec_s`
* `vec_cmp_d`: comparing & mask vector for `vec_d`
* `vec_cmp_t`: if `ASTER_DTYPE_DOUBLE` / `ASTER_DTYPE_FLOAT` is defined before including `aster.h`, `vec_cmp_t` == `vec_cmp_d` / `vec_cmp_s`



Function naming: `vec_<operation>_<s/d/t>`, suffix `s` is for `vec_f` data type, `d` is for `vec_d` data type, `t` is for `vec_t` type.

List of supported functions:

```c
vec_zero_*()            : Set all lanes of a vector to zero and return this vector
vec_set1_*(a)           : Set all lanes of a vector to value <a> and return this vector
vec_bcast_*(a)          : Set all lanes of a vector to value <a[0]> and return this vector
vec_load_*(a)           : Load a vector from an address <a> which must be aligned to required bits
vec_loadu_*(a)          : Load a vector from an address <a> which may not be aligned to required bits
vec_store_*(a, b)       : Store a vector <b> to an address <a> which must be aligned to required bits
vec_storeu_*(a, b)      : Store a vector <b> to an address <a> which may not be aligned to required bits
vec_add_*(a, b)         : Return lane-wise <a[i]> + <b[i]>
vec_sub_*(a, b)         : Return lane-wise <a[i]> - <b[i]>
vec_mul_*(a, b)         : Return lane-wise <a[i]> * <b[i]>
vec_div_*(a, b)         : Return lane-wise <a[i]> / <b[i]>
vec_abs_*(a)            : Return lane-wise abs(<a[i]>)
vec_sqrt_*(a)           : Return lane-wise sqrt(<a[i]>)
vec_fmadd_* (a, b, c)   : Return lane-wise Fused Multiply-Add            <a[i]> * <b[i]> + <c[i]>
vec_fnmadd_*(a, b, c)   : Return lane-wise Fused Negative Multiply-Add  -<a[i]> * <b[i]> + <c[i]>
vec_fmsub_* (a, b, c)   : Return lane-wise Fused Multiply-Sub intrinsic  <a[i]> * <b[i]> - <c[i]>
vec_max_*(a, b)         : Return lane-wise max(<a[i]>, <b[i]>)
vec_min_*(a, b)         : Return lane-wise min(<a[i]>, <b[i]>)
vec_cmp_eq_*(a, b)      : Return lane-wise if(<a[i]> == <b[i]>) compare vector
vec_cmp_ne_*(a, b)      : Return lane-wise if(<a[i]> != <b[i]>) compare vector
vec_cmp_lt_*(a, b)      : Return lane-wise if(<a[i]> <  <b[i]>) compare vector
vec_cmp_le_*(a, b)      : Return lane-wise if(<a[i]> <= <b[i]>) compare vector
vec_cmp_gt_*(a, b)      : Return lane-wise if(<a[i]> >  <b[i]>) compare vector
vec_cmp_ge_*(a, b)      : Return lane-wise if(<a[i]> >= <b[i]>) compare vector
vec_blend_*(a, b, m)    : Return lane-wise (<m[i]> == 1 ? <b[i]> : <a[i]>), a and b are data vectors, m is a compare vector
vec_reduce_add_*(a)     : Return a single value sum(<a[i]>)
vec_frsqrt_pf_*()       : Return scaling prefactor for vec_frsqrt_*()
vec_frsqrt_*(a)         : Return lane-wise fast reverse square root <a[i]> == 0 ? 0 : 1 / (sqrt(<a[i]>) * vec_frsqrt_pf_*())
```



The following math functions do not have corresponding native CPU instructions, but they are implemented in 3rd party libraries:

```c
vec_log_*(a)            : Return lane-wise ln(<a[i]>)
vec_log2_*(a)           : Return lane-wise log2(<a[i]>)
vec_log10_*(a)          : Return lane-wise log10(<a[i]>)
vec_exp_*(a)            : Return lane-wise exp(<a[i]>)
vec_exp2_*(a)           : Return lane-wise pow(2, <a[i]>)
vec_exp10_*(a)          : Return lane-wise pow(10, <a[i]>)
vec_pow_*(a, b)         : Return lane-wise pow(<a[i]>, <b[i]>)
vec_sin_*(a)            : Return lane-wise sin(<a[i]>)
vec_cos_*(a)            : Return lane-wise cos(<a[i]>)
vec_erf_*(a)            : Return lane-wise erf(<a[i]>)
```

* On x86, ASTER uses Intel SVML library or GCC libmvec when available
* On ARM, ASTER uses the ARM Performance Libraries (ARM PL) when available
* If 3rd party library is not available, ASTER falls back to for-loop implementation



Tested platforms:

| Processor                | Instruction Set | Operating System       | Compiler               |
| ------------------------ | --------------- | ---------------------- | ---------------------- |
| Intel Xeon Platinum 8160 | AVX-512         | CentOS 7.6, x64        | ICC 18.0.4             |
| Intel Xeon Gold 6226     | AVX-512         | CentOS 7.6, x64        | ICC 19.0.5             |
| Intel Xeon Phi 7210      | AVX-512         | CentOS 7.5, x64        | ICC 18.0.2             |
| Intel Core i7 8550U      | AVX2            | WSL2 Ubuntu 18.04, x64 | GCC 7.5.0              |
| Intel Xeon E5 2670       | AVX             | Ubuntu 18.04, x64      | GCC 7.5.0 & ICC 19.1.1 |
| Intel Core i5-7300U      | AVX2            | macOS 10.15.4, x64     | GCC 9.2.0              |
| AMD Threadripper 2950X   | AVX2            | WSL2 Ubuntu 20.04, x64 | GCC 9.3.0              |
| Fujitsu A64FX            | SVE-512 & NEON  | CentOS 8.3, aarch64    | GCC 10.2.0             |


