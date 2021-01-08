#ifndef __ASTER_H__
#define __ASTER_H__

#ifdef __AVX__
#include "avx_intrin_wrapper.h"
#endif

#ifdef __ARM_FEATURE_SVE
#include "sve_intrin_wrapper.h"
#endif

#if !defined(__AVX__) && !defined(__ARM_FEATURE_SVE)
#warning None of the following instruction set detected: AVX, AVX2, AVX512, SVE; using for loop simulated implementation.
#include "no_intrin_wrapper.h"
#endif

#endif
