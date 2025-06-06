//
// Created by BLD on 24-8-5.
//

#ifndef DEEPBIT_SPACE_H
#define DEEPBIT_SPACE_H
#pragma once

#include <queue>
#include <vector>
#include <algorithm>
#include <iostream>
#include <map>
#include <immintrin.h>

#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

#include "matrix.h"
#include "utils.h"
#include <random>

#if defined(LARGE_DATA)
const uint32_t B_QUERY = 4, numC = 65536;
#else
const uint32_t B_QUERY = 4, numC = 4096;
#endif
unsigned count_scan = 0, all_dist_count = 0;
float var_count = 5.0;

template<uint32_t D, uint32_t B>
class Space {
public:
    // ================================================================================================
    // ********************
    //   Binary Operation
    // ********************
    inline static uint32_t popcount(u_int64_t *d);

    inline static uint32_t ip_bin_bin(uint64_t *q, uint64_t *d);

    inline static uint32_t ip_byte_bin(uint64_t *q, uint64_t *d);

    inline static void transpose_bin(uint8_t *q, uint64_t *tq);

    // ================================================================================================
    inline static void range(float *q, float *c, float &vl, float &vr);

    inline static void
    quantize(uint8_t *result, float *q, float *c, float *u, float max_entry, float width, uint32_t &sum_q);

    inline static uint32_t sum(uint8_t *d);

    Space() {};

    ~Space() {};
};

class Disk_IO {
public:
    std::ifstream IO;
    unsigned D=0;

    void init_data_file(char *filename) {
        IO.open(filename, std::ios::binary);
        IO.read((char *) &D, sizeof(unsigned));
    }

    void single_disk_io(unsigned id, float *vec) {
        IO.seekg(std::ios::beg + (sizeof(float) * (D + 1)) * id + sizeof(float));
        IO.read((char *) vec, sizeof(float) * D);
    }

    Disk_IO() = default;

    ~Disk_IO() = default;
};


// ==============================================================
// inner product between binary strings
// ==============================================================
template<uint32_t D, uint32_t B>
inline uint32_t Space<D, B>::ip_bin_bin(uint64_t *q, uint64_t *d) {
    uint64_t ret = 0;
    for (int i = 0; i < B / 64; i++) {
        ret += __builtin_popcountll((*d) & (*q));
        q++;
        d++;
    }
    return ret;
}

// ==============================================================
// popcount (a.k.a, bitcount)
// ==============================================================
template<uint32_t D, uint32_t B>
inline uint32_t Space<D, B>::popcount(u_int64_t *d) {
    uint64_t ret = 0;
    for (int i = 0; i < B / 64; i++) {
        ret += __builtin_popcountll((*d));
        d++;
    }
    return ret;
}

// ==============================================================
// inner product between a decomposed byte string q
// and a binary string d
// ==============================================================
template<uint32_t D, uint32_t B>
uint32_t Space<D, B>::ip_byte_bin(uint64_t *q, uint64_t *d) {
    uint64_t ret = 0;
    for (int i = 0; i < B_QUERY; i++) {
        ret += (ip_bin_bin(q, d) << i);
        q += (B / 64);
    }
    return ret;
}

// ==============================================================
// decompose the quantized query vector into B_q binary vector
// ==============================================================
template<uint32_t D, uint32_t B>
void Space<D, B>::transpose_bin(uint8_t *q, uint64_t *tq) {
    for (int i = 0; i < B; i += 32) {
        __m256i v = _mm256_load_si256(reinterpret_cast<__m256i *>(q));
        v = _mm256_slli_epi32(v, (8 - B_QUERY));
        for (int j = 0; j < B_QUERY; j++) {
            uint32_t v1 = _mm256_movemask_epi8(v);
            v1 = reverseBits(v1);
            tq[(B_QUERY - j - 1) * (B / 64) + i / 64] |= ((uint64_t) v1 << ((i / 32 % 2 == 0) ? 32 : 0));
            v = _mm256_add_epi32(v, v);
        }
        q += 32;
    }
}

// ==============================================================
// compute the min and max value of the entries of q
// ==============================================================
template<uint32_t D, uint32_t B>
void Space<D, B>::range(float *q, float *c, float &vl, float &vr) {
    vl = +1e20;
    vr = -1e20;
    for (int i = 0; i < B; i++) {
        float tmp = (*q) - (*c);
        if (tmp < vl)vl = tmp;
        if (tmp > vr)vr = tmp;
        q++;
        c++;
    }
}

// ==============================================================
// quantize the query vector with uniform scalar quantization
// ==============================================================
template<uint32_t D, uint32_t B>
void Space<D, B>::quantize(uint8_t *result, float *q, float *c, float *u, float vl, float width, uint32_t &sum_q) {
    float one_over_width = 1.0 / width;
    uint8_t *ptr_res = result;
    uint32_t sum = 0;
    for (int i = 0; i < B; i++) {
        (*ptr_res) = (uint8_t) ((((*q) - (*c)) - vl) * one_over_width + (*u));
        sum += (*ptr_res);
        q++;
        c++;
        ptr_res++;
        u++;
    }
    sum_q = sum;
}

// The implementation is based on https://github.com/nmslib/hnswlib/blob/master/hnswlib/space_ip.h
template<uint32_t L>
inline float sqr_dist(float *d, float *q) {
    float PORTABLE_ALIGN32 TmpRes[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    constexpr uint32_t num_blk16 = L >> 4;
    constexpr uint32_t l = L & 0b1111;

    __m256 diff, v1, v2;
    __m256 sum = _mm256_set1_ps(0);
    for (int i = 0; i < num_blk16; i++) {
        v1 = _mm256_loadu_ps(d);
        v2 = _mm256_loadu_ps(q);
        d += 8;
        q += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        v1 = _mm256_loadu_ps(d);
        v2 = _mm256_loadu_ps(q);
        d += 8;
        q += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }
    for (int i = 0; i < l / 8; i++) {
        v1 = _mm256_loadu_ps(d);
        v2 = _mm256_loadu_ps(q);
        d += 8;
        q += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }
    _mm256_store_ps(TmpRes, sum);

    float ret = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

    for (int i = 0; i < l % 8; i++) {
        float tmp = (*q) - (*d);
        ret += tmp * tmp;
        d++;
        q++;
    }
    return ret;
}

template<uint32_t L>
inline float ip_sim(float *d, float *q) {
    float PORTABLE_ALIGN32 TmpRes[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    constexpr uint32_t num_blk16 = L >> 4;
    constexpr uint32_t l = L & 0b1111;

    __m256 v1, v2;
    __m256 sum = _mm256_set1_ps(0);
    for (int i = 0; i < num_blk16; i++) {
        v1 = _mm256_loadu_ps(d);
        v2 = _mm256_loadu_ps(q);
        d += 8;
        q += 8;
        sum = _mm256_add_ps(sum, _mm256_mul_ps(v1, v2));

        v1 = _mm256_loadu_ps(d);
        v2 = _mm256_loadu_ps(q);
        d += 8;
        q += 8;
        sum = _mm256_add_ps(sum, _mm256_mul_ps(v1, v2));
    }
    for (int i = 0; i < l / 8; i++) {
        v1 = _mm256_loadu_ps(d);
        v2 = _mm256_loadu_ps(q);
        d += 8;
        q += 8;
        sum = _mm256_add_ps(sum, _mm256_mul_ps(v1, v2));
    }
    _mm256_store_ps(TmpRes, sum);

    float ret = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

    for (int i = 0; i < l % 8; i++) {
        ret += (*q) * (*d);
        d++;
        q++;
    }
    return ret;
}

template<uint32_t L>
inline float mask_last_ip_sim127(float *d, float *q) {
    float PORTABLE_ALIGN32 TmpRes[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    constexpr uint32_t num_blk16 = L >> 4;

    __m256 v1, v2;
    __m256 sum = _mm256_set1_ps(0);
    for (int i = 0; i < num_blk16; i++) {
        v1 = _mm256_loadu_ps(d);
        v2 = _mm256_loadu_ps(q);
        d += 8;
        q += 8;
        sum = _mm256_add_ps(sum, _mm256_mul_ps(v1, v2));

        v1 = _mm256_loadu_ps(d);
        v2 = _mm256_loadu_ps(q);
        d += 8;
        q += 8;
        sum = _mm256_add_ps(sum, _mm256_mul_ps(v1, v2));
    }

    v1 = _mm256_loadu_ps(d);
    v2 = _mm256_loadu_ps(q);
    d += 8;
    q += 8;
    sum = _mm256_add_ps(sum, _mm256_mul_ps(v1, v2));

    __m256i mask = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, -1, 0);
    v1 = _mm256_maskload_ps(d, mask);
    v2 = _mm256_loadu_ps(q);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(v1, v2));

    _mm256_store_ps(TmpRes, sum);

    float ret = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

    return ret;
}

template<uint32_t L>
float vec_sqr(const float *q) {
    float sqr = 0;
    for (int i = 0; i < L; i++) {
        sqr += q[i] * q[i];
    }
    return sqr;
}

float naive_l2_dist(float *q, float *p, int d) {
    float sqr = 0;
    for (int i = 0; i < d; i++) {
        sqr += (q[i] - p[i]) * (q[i] - p[i]);
    }
    return sqr;
}

float naive_ip_dist(float *q, float *p, int d) {
    float sqr = 0;
    for (int i = 0; i < d; i++) {
        sqr += p[i] * q[i];
    }
    return sqr;
}


#endif //DEEPBIT_SPACE_H
