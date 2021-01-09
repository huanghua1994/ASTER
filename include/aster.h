#ifndef __ASTER_H__
#define __ASTER_H__

#if defined(__AVX__)
#include "avx_intrin_wrapper.h"
#endif

#if defined(__ARM_NEON) && !defined(__ARM_FEATURE_SVE)
#include "neon_intrin_wrapper.h"
#endif

#if defined(__ARM_FEATURE_SVE)
#include "sve_intrin_wrapper.h"
#endif

#if !defined(__AVX__) && !defined(__ARM_NEON) && !defined(__ARM_FEATURE_SVE)
#warning None of the following instruction set detected: AVX, AVX2, AVX512, NEON, SVE; using for loop simulated implementation.
#include "no_intrin_wrapper.h"
#endif

#endif
