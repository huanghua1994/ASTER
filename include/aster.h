#ifndef __ASTER_H__
#define __ASTER_H__

#if defined(__AVX__) && !defined(__AVX512F__)
#include "avx_intrin_wrapper.h"
#endif

#if defined(USE_AVX)
#include "avx_intrin_wrapper.h"
#endif

#if defined(__AVX512F__) && !defined(USE_AVX)
#include "avx512_intrin_wrapper.h"
#endif

#if defined(__ARM_NEON) && !defined(__ARM_FEATURE_SVE)
#include "neon_intrin_wrapper.h"
#endif

#if defined(USE_ASIMD)
#include "neon_intrin_wrapper.h"
#endif

#if defined(__ARM_FEATURE_SVE) && !defined(USE_ASIMD)
#include "sve_intrin_wrapper.h"
#endif

#if !defined(__AVX__) && !defined(__ARM_NEON) && !defined(__ARM_FEATURE_SVE)
#warning None of the following instruction set detected: AVX, AVX2, AVX512, NEON, SVE; using for loop simulated implementation.
#include "no_intrin_wrapper.h"
#endif

#include "dtype_wrapper.h"

#endif
