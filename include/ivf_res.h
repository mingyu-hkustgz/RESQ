//
// Created by BLD on 24-8-31.
//

#ifndef DEEPBIT_IVF_RES_H
#define DEEPBIT_IVF_RES_H

#include <queue>
#include <vector>
#include <algorithm>
#include <map>
#include "matrix.h"
#include "utils.h"
#include "space.h"
#include "fast_scan.h"

template<uint32_t D, uint32_t B>
class IVFRES {
private:
public:
    struct Factor {
        float sqr_x;
        float error;
        float factor_ppc;
        float factor_ip;
    };

    Factor *fac;
    static constexpr float fac_norm = const_sqrt(1.0 * B);
    static constexpr float max_x1 = 1.9 / const_sqrt(1.0 * B - 1.0);

    static Space<D, B> space;

    uint32_t N;                       // the number of data vectors
    uint32_t C;                       // the number of clusters
    uint32_t RD;                      // the residual dimensionality

    uint32_t *start;                  // the start point of a cluster
    uint32_t *packed_start;           // the start point of a cluster (packed with batch of 32)
    uint32_t *len;                    // the length of a cluster
    uint32_t *id;                     // N of size_t the ids of the objects in a cluster
    float *dist_to_c;                // N of floats distance to the centroids (not the squared distance)
    float *u;                        // B of floats random numbers sampled from the uniform distribution [0,1]
    float *var_;                     // variance of each dimension
    float *mean_;                    // mean of each dimension

    uint64_t *binary_code;           // (B / 64) * N of 64-bit uint64_t
    uint8_t *packed_code;            // packed code with the batch size of 32 vectors


    float *x0;                       // N of floats in the Random Net algorithm
    float *centroid;                 // N * B floats (not N * D), note that the centroids should be randomized
    float *data;                     // N * D floats, note that the datas are not randomized
    float *res_data;                 // N * RD floats,

    IVFRES();

    IVFRES(const Matrix<float> &X, const Matrix<float> &_centroids, const Matrix<float> &dist_to_centroid,
           const Matrix<float> &_x0, const Matrix<uint32_t> &cluster_id, const Matrix<uint64_t> &binary);

    ~IVFRES();

    ResultHeap search(float *query, float *rd_query, uint32_t k, uint32_t nprobe,
                      float distK = std::numeric_limits<float>::max()) const;

    static void fast_scan(ResultHeap &KNNs, float &distK, uint32_t k,
                          uint8_t *LUT, uint8_t *packed_code, uint32_t len, Factor *ptr_fac,
                          const float sqr_y, const float res_sqr_y, const float res_error, const float vl,
                          const float width, const float sumq,
                          float *query, float *data, uint32_t *id);

    void save(char *filename);

    void load(char *filename);
};


template<uint32_t D, uint32_t B>
void IVFRES<D, B>::fast_scan(ResultHeap &KNNs, float &distK, uint32_t k,
                             uint8_t *LUT, uint8_t *packed_code, uint32_t len, Factor *ptr_fac,
                             const float sqr_y, const float res_sqr_y, const float res_error, const float vl,
                             const float width, const float sumq,
                             float *query, float *data, uint32_t *id) {

    for (int i = 0; i < B / 4 * 16; i++)LUT[i] *= 2;

    float y = std::sqrt(sqr_y);

    constexpr uint32_t SIZE = 32;
    uint32_t it = len / SIZE;
    uint32_t remain = len - it * SIZE;
    uint32_t nblk_remain = (remain + 31) / 32;

    while (it--) {
        float low_dist[SIZE];
        float *ptr_low_dist = &low_dist[0];
        uint16_t PORTABLE_ALIGN32 result[SIZE];
        accumulate<B>((SIZE / 32), packed_code, LUT, result);
        packed_code += SIZE * B / 8;

        for (int i = 0; i < SIZE; i++) {
            float tmp_dist = (ptr_fac->sqr_x) + sqr_y + res_sqr_y + ptr_fac->factor_ppc * vl +
                             (result[i] - sumq) * (ptr_fac->factor_ip) * width;
            float error_bound = y * (ptr_fac->error) + res_error;
            *ptr_low_dist = tmp_dist - error_bound;
            ptr_fac++;
            ptr_low_dist++;
#ifdef COUNT_SCAN
            all_dist_count++;
#endif
        }
        ptr_low_dist = &low_dist[0];
        for (int j = 0; j < SIZE; j++) {
            if (*ptr_low_dist < distK) {

                float gt_dist = sqr_dist<D>(query, data);
#ifdef COUNT_SCAN
                count_scan++;
#endif
                if (gt_dist < distK) {
                    KNNs.emplace(gt_dist, *id);
                    if (KNNs.size() > k) KNNs.pop();
                    if (KNNs.size() == k)distK = KNNs.top().first;
                }
            }
            data += D;
            ptr_low_dist++;
            id++;
        }
    }

    {
        float low_dist[SIZE];
        float *ptr_low_dist = &low_dist[0];
        uint16_t PORTABLE_ALIGN32 result[SIZE];
        accumulate<B>(nblk_remain, packed_code, LUT, result);

        for (int i = 0; i < remain; i++) {
            float tmp_dist = (ptr_fac->sqr_x) + sqr_y + res_sqr_y + ptr_fac->factor_ppc * vl +
                             (result[i] - sumq) * ptr_fac->factor_ip * width;
            float error_bound = y * (ptr_fac->error) + res_error;
#ifdef COUNT_SCAN
            all_dist_count++;
#endif
            // ***********************************************************************************************
            *ptr_low_dist = tmp_dist - error_bound;
            ptr_fac++;
            ptr_low_dist++;
        }
        ptr_low_dist = &low_dist[0];
        for (int i = 0; i < remain; i++) {
            if (*ptr_low_dist < distK) {

                float gt_dist = sqr_dist<D>(query, data);
#ifdef COUNT_SCAN
                count_scan++;
#endif
                if (gt_dist < distK) {
                    KNNs.emplace(gt_dist, *id);
                    if (KNNs.size() > k) KNNs.pop();
                    if (KNNs.size() == k)distK = KNNs.top().first;
                }
            }
            data += D;
            ptr_low_dist++;
            id++;
        }
    }
}

// search impl
template<uint32_t D, uint32_t B>
ResultHeap
IVFRES<D, B>::search(float *query, float *rd_query, uint32_t k, uint32_t nprobe, float distK) const {
    // The default value of distK is +inf
    ResultHeap KNNs;
    // ===========================================================================================================
    // Compute the residual error bound
    float res_error = 0;
    for (int i = B; i < D; i++) {
        res_error += var_[i] * (query[i] - mean_[i]) * (query[i] - mean_[i]);
    }

#ifdef RESIDUAL_SPLIT
    float p_res_error = res_error - var_[B] * (query[B] - mean_[B]) * (query[B] - mean_[B]);
    p_res_error = 4.0f * std::sqrt(p_res_error);
#endif

    res_error = 4.0f * std::sqrt(res_error);

    // ===========================================================================================================
    // Find out the nearest N_{probe} centroids to the query vector.
    Result centroid_dist[numC];
    float *ptr_c = centroid;
    float res_sqr_y = ip_sim<D - B>(query + B, query + B);
    for (int i = 0; i < C; i++) {
        centroid_dist[i].first = sqr_dist<B>(rd_query, ptr_c);
        centroid_dist[i].second = i;
        ptr_c += B;
    }
    std::partial_sort(centroid_dist, centroid_dist + nprobe, centroid_dist + numC);

    // ===========================================================================================================
    // Scan the first nprobe clusters.
    Result *ptr_centroid_dist = (&centroid_dist[0]);
    uint8_t  PORTABLE_ALIGN64 byte_query[B];

    for (int pb = 0; pb < nprobe; pb++) {
        uint32_t c = ptr_centroid_dist->second;
        float sqr_y = ptr_centroid_dist->first;
        ptr_centroid_dist++;

        // =======================================================================================================
        // Preprocess the residual query and the quantized query
        float vl, vr;
        space.range(rd_query, centroid + c * B, vl, vr);
        float width = (vr - vl) / ((1 << B_QUERY) - 1);
        uint32_t sum_q = 0;
        space.quantize(byte_query, rd_query, centroid + c * B, u, vl, width, sum_q);

        uint8_t PORTABLE_ALIGN32 LUT[B / 4 * 16];
        pack_LUT<B>(byte_query, LUT);

        fast_scan(KNNs, distK, k, \
                LUT, packed_code + packed_start[c], len[c], fac + start[c], \
                sqr_y, res_sqr_y, res_error, vl, width, sum_q, \
                query, data + start[c] * D, id + start[c]);

    }
    return KNNs;
}


// ==============================================================================================================================
// Save and Load Functions
template<uint32_t D, uint32_t B>
void IVFRES<D, B>::save(char *filename) {
    std::ofstream output(filename, std::ios::binary);

    uint32_t d = D;
    uint32_t b = B;
    output.write((char *) &N, sizeof(uint32_t));
    output.write((char *) &d, sizeof(uint32_t));
    output.write((char *) &C, sizeof(uint32_t));
    output.write((char *) &b, sizeof(uint32_t));

    output.write((char *) start, C * sizeof(uint32_t));
    output.write((char *) len, C * sizeof(uint32_t));
    output.write((char *) id, N * sizeof(uint32_t));
    output.write((char *) dist_to_c, N * sizeof(float));
    output.write((char *) x0, N * sizeof(float));
    output.write((char *) var_, D * sizeof(float));
    output.write((char *) mean_, D * sizeof(float));

    output.write((char *) centroid, C * B * sizeof(float));
    output.write((char *) data, N * D * sizeof(float));
    output.write((char *) binary_code, N * B / 64 * sizeof(uint64_t));

    output.close();
    std::cerr << "Saved!" << std::endl;
}

// load impl
template<uint32_t D, uint32_t B>
void IVFRES<D, B>::load(char *filename) {
    std::ifstream input(filename, std::ios::binary);
    //std::cerr << filename << std::endl;

    if (!input.is_open())
        throw std::runtime_error("Cannot open file");

    uint32_t d;
    uint32_t b;
    input.read((char *) &N, sizeof(uint32_t));
    input.read((char *) &d, sizeof(uint32_t));
    input.read((char *) &C, sizeof(uint32_t));
    input.read((char *) &b, sizeof(uint32_t));

    assert(d == D);
    assert(b == B);

    u = new float[B];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> uniform(0.0, 1.0);
    for (int i = 0; i < B; i++)u[i] = uniform(gen);

    centroid = new float[C * B];
    data = new float[N * D];

    binary_code = static_cast<uint64_t *>(aligned_alloc(256, N * B / 64 * sizeof(uint64_t)));

    start = new uint32_t[C];
    len = new uint32_t[C];
    id = new uint32_t[N];
    dist_to_c = new float[N];
    x0 = new float[N];

    fac = new Factor[N];

    var_ = new float [D];
    mean_ = new float [D];

    input.read((char *) start, C * sizeof(uint32_t));
    input.read((char *) len, C * sizeof(uint32_t));
    input.read((char *) id, N * sizeof(uint32_t));
    input.read((char *) dist_to_c, N * sizeof(float));
    input.read((char *) x0, N * sizeof(float));
    input.read((char *) var_, D * sizeof(float));
    input.read((char *) mean_, D * sizeof(float));

    input.read((char *) centroid, C * B * sizeof(float));
    input.read((char *) data, N * D * sizeof(float));
    input.read((char *) binary_code, N * B / 64 * sizeof(uint64_t));

    packed_start = new uint32_t[C];
    int cur = 0;
    for (int i = 0; i < C; i++) {
        packed_start[i] = cur;
        cur += (len[i] + 31) / 32 * 32 * B / 8;
    }
    packed_code = static_cast<uint8_t *>(aligned_alloc(32, cur * sizeof(uint8_t)));
    for (int i = 0; i < C; i++) {
        pack_codes<B>(binary_code + start[i] * (B / 64), len[i], packed_code + packed_start[i]);
    }
    double ave_error = 0;
    for (int i = 0; i < N; i++) {
        long double x_x0 = (long double) dist_to_c[i] / x0[i];
        fac[i].sqr_x = dist_to_c[i] * dist_to_c[i] +
                       ip_sim<D - B>(data + i * D + B, data + i * D + B); // add the residual dim norm
        fac[i].error = 2 * max_x1 * std::sqrt(x_x0 * x_x0 - dist_to_c[i] * dist_to_c[i]);
        ave_error += fac[i].error;
        fac[i].factor_ppc = -2 / fac_norm * x_x0 * ((float) space.popcount(binary_code + i * B / 64) * 2 - B);
        fac[i].factor_ip = -2 / fac_norm * x_x0;
    }
    ave_error /= N;
    std::cerr << "ave error:: " << ave_error << std::endl;
    input.close();
}


// ==============================================================================================================================
// Construction and Deconstruction Functions
template<uint32_t D, uint32_t B>
IVFRES<D, B>::IVFRES() {
    N = C = 0;
    start = len = id = NULL;
    x0 = dist_to_c = centroid = data = NULL;
    binary_code = NULL;
    fac = NULL;
    u = NULL;
}

template<uint32_t D, uint32_t B>
IVFRES<D, B>::IVFRES(const Matrix<float> &X, const Matrix<float> &_centroids, const Matrix<float> &dist_to_centroid,
                     const Matrix<float> &_x0, const Matrix<uint32_t> &cluster_id, const Matrix<uint64_t> &binary) {
    fac = NULL;
    u = NULL;

    N = X.n;
    C = _centroids.n;
    RD = D - B + 1;
    // check uint64_t
    assert(B % 64 == 0);

    start = new uint32_t[C];
    len = new uint32_t[C];
    id = new uint32_t[N];
    dist_to_c = new float[N];
    x0 = new float[N];

    memset(len, 0, C * sizeof(uint32_t));
    for (int i = 0; i < N; i++)len[cluster_id.data[i]]++;
    int sum = 0;
    for (int i = 0; i < C; i++) {
        start[i] = sum;
        sum += len[i];
    }
    for (int i = 0; i < N; i++) {
        id[start[cluster_id.data[i]]] = i;
        dist_to_c[start[cluster_id.data[i]]] = dist_to_centroid.data[i];
        x0[start[cluster_id.data[i]]] = _x0.data[i];
        start[cluster_id.data[i]]++;
    }
    for (int i = 0; i < C; i++) {
        start[i] -= len[i];
    }

    centroid = new float[C * B];
#ifdef RESIDUAL_SPLIT
    data = new float[N * B];
    res_data = new float[N * RD];
#else
    data = new float[N * D];
#endif
    binary_code = new uint64_t[N * B / 64];

    std::memcpy(centroid, _centroids.data, C * B * sizeof(float));
    float *data_ptr = data;
    float *res_ptr = res_data;
    uint64_t *binary_code_ptr = binary_code;
    for (int i = 0; i < N; i++) {
        int x = id[i];
#ifdef RESIDUAL_SPLIT
        data_ptr[0] = ip_sim<D>(X.data + x * X.d, X.data + x * X.d);
        std::memcpy(data_ptr + 1, X.data + x * X.d, (B - 1) * sizeof(float));
        std::memcpy(res_ptr, X.data + x * X.d + D, RD * sizeof(float));
        data_ptr += B;
        res_ptr += RD;
#else
        std::memcpy(data_ptr, X.data + x * X.d, D * sizeof(float));
        data_ptr += D;
#endif
        std::memcpy(binary_code_ptr, binary.data + x * (B / 64), (B / 64) * sizeof(uint64_t));
        binary_code_ptr += B / 64;
    }

    std::vector<double> mean(D);
#pragma omp parallel for
    for (int j = 0; j < D; j++) {
        for (int i = 0; i < N; i++) {
            mean[j] += X.data[i * D + j];
        }
    }
    mean_ = new float[D];
    for (int j = 0; j < D; j++) mean_[j] = mean[j] / N;

    std::vector<double> variance(D);
#pragma omp parallel for
    for (int j = 0; j < D; j++) {
        for (int i = 0; i < N; i++) {
            variance[j] += (X.data[i * D + j] - mean_[j]) * (X.data[i * D + j] - mean_[j]);
        }
    }
    var_ = new float[D];
    for (int j = 0; j < D; j++) var_[j] = variance[j] / N;
}

template<uint32_t D, uint32_t B>
IVFRES<D, B>::~IVFRES() {
    if (id != NULL) delete[] id;
    if (dist_to_c != NULL) delete[] dist_to_c;
    if (len != NULL) delete[] len;
    if (start != NULL) delete[] start;
    if (x0 != NULL) delete[] x0;
    if (data != NULL) delete[] data;
    if (fac != NULL) delete[] fac;
    if (u != NULL) delete[] u;
    if (var_ != NULL) delete[] var_;
    if (mean_ != NULL) delete[] mean_;
    if (binary_code != NULL) std::free(binary_code);
    if (centroid != NULL) std::free(centroid);
}

#endif //DEEPBIT_IVF_RES_H
