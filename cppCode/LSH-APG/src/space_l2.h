#pragma once
#include <atomic>
//#include "fastL2_ip.h"
extern int _G_COST;
template<typename MTYPE>
using DISTFUNC = MTYPE(*)(const void*, const void*, const void*);


static float
    L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    ++_G_COST;
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);

    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        float t = *pVect1 - *pVect2;
        pVect1++;
        pVect2++;
        res += t * t;
    }
    return (res);
}

#if defined(USE_AVX)

// Favor using AVX if available.
static float
    L2SqrSIMD16Ext(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    ++_G_COST;
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty16 = qty >> 4;

    const float* pEnd1 = pVect1 + (qty16 << 4);

    __m256 diff, v1, v2;
    __m256 sum = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }

    _mm256_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
}

#elif defined(USE_SSE)

static float
    L2SqrSIMD16Ext(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    ++_G_COST;
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty16 = qty >> 4;

    const float* pEnd1 = pVect1 + (qty16 << 4);

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }

    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}
#endif

#if defined(USE_SSE) || defined(USE_AVX)
static float
    L2SqrSIMD16ExtResiduals(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    ++_G_COST;
    size_t qty = *((size_t*)qty_ptr);
    size_t qty16 = qty >> 4 << 4;
    float res = L2SqrSIMD16Ext(pVect1v, pVect2v, &qty16);
    float* pVect1 = (float*)pVect1v + qty16;
    float* pVect2 = (float*)pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = L2Sqr(pVect1, pVect2, &qty_left);
    return (res + res_tail);
}
#endif


#ifdef USE_SSE
static float
    L2SqrSIMD4Ext(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    ++_G_COST;
    float PORTABLE_ALIGN32 TmpRes[8];
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);


    size_t qty4 = qty >> 2;

    const float* pEnd1 = pVect1 + (qty4 << 2);

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }
    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

static float
    L2SqrSIMD4ExtResiduals(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    ++_G_COST;
    size_t qty = *((size_t*)qty_ptr);
    size_t qty4 = qty >> 2 << 2;

    float res = L2SqrSIMD4Ext(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    float* pVect1 = (float*)pVect1v + qty4;
    float* pVect2 = (float*)pVect2v + qty4;
    float res_tail = L2Sqr(pVect1, pVect2, &qty_left);

    return (res + res_tail);
}
#endif


#include <mutex>
#include <thread>
#include <atomic>
#include <vector>
// Multithreaded executor
// The helper function copied from python_bindings/bindings.cpp (and that itself is copied from nmslib)
// An alternative is using #pragme omp parallel for or any other C++ threading
template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
	if (numThreads <= 0) {
		numThreads = std::thread::hardware_concurrency();
	}

	if (numThreads == 1) {
		for (size_t id = start; id < end; id++) {
			fn(id, 0);
		}
	}
	else {
		std::vector<std::thread> threads;
		std::atomic<size_t> current(start);

		// keep track of exceptions in threads
		// https://stackoverflow.com/a/32428427/1713196
		std::exception_ptr lastException = nullptr;
		std::mutex lastExceptMutex;

		for (size_t threadId = 0; threadId < numThreads; ++threadId) {
			threads.push_back(std::thread([&, threadId] {
				while (true) {
					size_t id = current.fetch_add(1);

					if (id >= end) {
						break;
					}

					try {
						fn(id, threadId);
					}
					catch (...) {
						std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
						lastException = std::current_exception();
						/*
						 * This will work even when current is the largest value that
						 * size_t can fit, because fetch_add returns the previous value
						 * before the increment (what will result in overflow
						 * and produce 0 instead of current + 1).
						 */
						current = end;
						break;
					}
				}
				}));
		}
		for (auto& thread : threads) {
			thread.join();
		}
		if (lastException) {
			std::rethrow_exception(lastException);
		}
	}
}
