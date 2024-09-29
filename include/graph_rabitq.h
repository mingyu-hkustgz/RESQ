//
// Created by BLD on 24-9-23.
//

#ifndef DEEPBIT_GRAPH_RABITQ_H
#define DEEPBIT_GRAPH_RABITQ_H

#include "fast_scan.h"
#include "space.h"
#include "utils.h"
#include "matrix.h"
#include "graph.h"
#include <boost/dynamic_bitset.hpp>

template<uint32_t D, uint32_t B>
class GraphRabit {
private:
protected:
    typedef std::vector<std::vector<unsigned>> CompactGraph;
    typedef std::vector<SimpleNeighbors> LockGraph;
public:
    struct Factor {
        float sqr_x;
        float error;
        float factor_ppc;
        float factor_ip;
    };

    GraphRabit();

    GraphRabit(const Matrix<float> &X);

    GraphRabit(const Matrix<float> &X, const Matrix<float> &_centroids,
               const Matrix<float> &dist_to_centroid, const Matrix<float> &_x0,
               const Matrix<uint64_t> &binary);

    void Search(float *query, int k, int L, ResultHeap &KNNs) const;

    void SearchRabit(float *query, float *rd_query, int k, int L, ResultHeap &KNNs) const;

    void LoadGraph(const char *filename);

    void Save(const char *filename);

    void Load(const char *filename);

    CompactGraph final_graph_;
    unsigned width{};
    unsigned ep_{}, nd_; //not in use
    std::vector<unsigned> eps_;

    Factor *fac;
    static constexpr float fac_norm = const_sqrt(1.0 * B);
    static constexpr float max_x1 = 1.9 / const_sqrt(1.0 * B - 1.0);

    static Space<D, B> space;

    float *data_;
    float *centroid_;
    float *x0;                       // N of floats of point-centroid inner product
    float *u;                        // B of floats random numbers sampled from the uniform distribution [0,1]
    uint64_t *binary_code;           // (B / 64) * N of 64-bit uint64_t
};


// ==============================================================================================================================
// Construction and Deconstruction Functions
template<uint32_t D, uint32_t B>
GraphRabit<D, B>::GraphRabit() {
    nd_ = 0;
    x0 = centroid_ = data_ = NULL;
    binary_code = NULL;
    fac = NULL;
    u = NULL;
}

template<uint32_t D, uint32_t B>
GraphRabit<D, B>::GraphRabit(const Matrix<float> &X) {
    nd_ = X.n;
    data_ = new float[nd_ * D];
    std::memcpy(data_, X.data, nd_ * D * sizeof(float));
}

template<uint32_t D, uint32_t B>
GraphRabit<D, B>::GraphRabit(const Matrix<float> &X, const Matrix<float> &centroid,
                             const Matrix<float> &x2, const Matrix<float> &_x0,
                             const Matrix<uint64_t> &binary) {
    nd_ = X.n;
    data_ = new float[nd_ * D];
    x0 = new float[nd_];
    centroid_ = new float[B];
    binary_code = new uint64_t[nd_ * B / 64];
    fac = new Factor[nd_];

    std::memcpy(data_, X.data, nd_ * D * sizeof(float));
    std::memcpy(centroid_, centroid.data, B * sizeof(float));
    std::memcpy(x0, _x0.data, nd_ * sizeof(float));
    std::memcpy(binary_code, binary.data, nd_ * (B / 64) * sizeof(uint64_t));

    u = new float[B];
#if defined(RANDOM_QUERY_QUANTIZATION)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> uniform(0.0, 1.0);
    for(int i=0;i<B;i++)u[i] = uniform(gen);
#else
    for (int i = 0; i < B; i++) u[i] = 0.5;
#endif

    double ave_error = 0;
    for (int i = 0; i < nd_; i++) {
        long double x_x0 = (long double) x2.data[i] / x0[i];
        fac[i].sqr_x = x2.data[i] * x2.data[i];
        fac[i].error = 2 * max_x1 * std::sqrt(x_x0 * x_x0 - x2.data[i] * x2.data[i]);
        ave_error += fac[i].error;
        fac[i].factor_ppc = -2 / fac_norm * x_x0 * ((float) space.popcount(binary_code + i * B / 64) * 2 - B);
        fac[i].factor_ip = -2 / fac_norm * x_x0;
    }
    ave_error /= nd_;
    std::cerr << "ave error:: " << ave_error << std::endl;
}


template<uint32_t D, uint32_t B>
void GraphRabit<D, B>::LoadGraph(const char *filename) {
    std::ifstream in(filename, std::ios::binary);
    in.read((char *) &width, sizeof(unsigned));
    unsigned n_ep = 0;
    in.read((char *) &n_ep, sizeof(unsigned));
    eps_.resize(n_ep);
    in.read((char *) eps_.data(), n_ep * sizeof(unsigned));
    // width=100;
    unsigned cc = 0;
    while (!in.eof()) {
        unsigned k;
        in.read((char *) &k, sizeof(unsigned));
        if (in.eof()) break;
        cc += k;
        std::vector<unsigned> tmp(k);
        in.read((char *) tmp.data(), k * sizeof(unsigned));
        final_graph_.push_back(tmp);
    }
    cc /= nd_;
    std::cerr << "Average Degree = " << cc << std::endl;
}

template<uint32_t D, uint32_t B>
void GraphRabit<D, B>::Search(float *query, int K, int L, ResultHeap &KNNs) const {
    std::vector<Neighbor> retset(L + 1);
    std::vector<unsigned> init_ids(L);
    boost::dynamic_bitset<> flags{nd_, 0};
    std::mt19937 rng(rand());
    GenRandom(rng, init_ids.data(), L, (unsigned) nd_);
    assert(eps_.size() < L);
    float *x = data_;
    for (unsigned i = 0; i < eps_.size(); i++) {
        init_ids[i] = eps_[i];
    }

    for (unsigned i = 0; i < L; i++) {
        unsigned id = init_ids[i];
        float dist = sqr_dist<D>(query, x + id * D);
        retset[i] = Neighbor(id, dist, true);
        flags[id] = true;
    }

    std::sort(retset.begin(), retset.begin() + L);
    int k = 0;
    while (k < (int) L) {
        int nk = L;

        if (retset[k].flag) {
            retset[k].flag = false;
            unsigned n = retset[k].id;

            for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
                unsigned id = final_graph_[n][m];
                if (flags[id]) continue;
                flags[id] = true;
                float dist = sqr_dist<D>(query, x + id * D);
                if (dist >= retset[L - 1].distance) continue;
                Neighbor nn(id, dist, true);
                int r = InsertIntoPool(retset.data(), L, nn);

                if (r < nk) nk = r;
            }
        }
        if (nk <= k)
            k = nk;
        else
            ++k;
    }
    for (size_t i = 0; i < K; i++) {
        KNNs.emplace(retset[i].distance, retset[i].id);
    }
}


template<uint32_t D, uint32_t B>
void GraphRabit<D, B>::SearchRabit(float *query, float *rd_query, int K, int L, ResultHeap &KNNs) const {
    std::vector<Neighbor> retset(L + 1);
    std::vector<unsigned> init_ids(L);
    boost::dynamic_bitset<> flags{nd_, 0};
    std::mt19937 rng(rand());
    GenRandom(rng, init_ids.data(), L, (unsigned) nd_);
    assert(eps_.size() < L);
    for (unsigned i = 0; i < eps_.size(); i++) {
        init_ids[i] = eps_[i];
    }

    for (unsigned i = 0; i < L; i++) {
        unsigned id = init_ids[i];
        float dist = sqr_dist<D>(query, data_ + id * D);
        retset[i] = Neighbor(id, dist, true);
        flags[id] = true;
    }

    uint8_t  PORTABLE_ALIGN64 byte_query[B];
    // =======================================================================================================
    // Preprocess the residual query and the quantized query
    float vl, vr;
    space.range(rd_query, centroid_, vl, vr);
    float width = (vr - vl) / ((1 << B_QUERY) - 1);
    uint32_t sum_q = 0;
    space.quantize(byte_query, rd_query, centroid_, u, vl, width, sum_q);
    auto sumq = (float) sum_q;//Continuity [important]
    uint64_t PORTABLE_ALIGN32 quant_query[B_QUERY * B / 64];
    memset(quant_query, 0, sizeof(quant_query));
    space.transpose_bin(byte_query, quant_query);

    float sqr_y = sqr_dist<B>(rd_query, centroid_);
    float y = std::sqrt(sqr_y);

    std::sort(retset.begin(), retset.begin() + L);
    int k = 0;
    while (k < (int) L) {
        int nk = L;

        if (retset[k].flag) {
            retset[k].flag = false;
            unsigned n = retset[k].id;

            for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
                unsigned id = final_graph_[n][m];
                if (flags[id]) continue;
                flags[id] = true;
#ifdef COUNT_SCAN
                all_dist_count++;
#endif
                float tmp_dist = (fac[id].sqr_x) + sqr_y + fac[id].factor_ppc * vl +
                                 (space.ip_byte_bin(quant_query, binary_code + id * (B / 64)) * 2 - sumq) *
                                 (fac[id].factor_ip) * width;
                float error_bound = y * (fac[id].error);
                if (tmp_dist - error_bound < retset[K - 1].distance) {
#ifdef COUNT_SCAN
                    count_scan++;
#endif
                    float dist = sqr_dist<D>(query, data_ + id * D);
                    Neighbor nn(id, dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);
                    if (r < nk) nk = r;
                } else {
                    if (tmp_dist >= retset[L - 1].distance) continue;
                    Neighbor nn(id, tmp_dist, true);
                    int r = InsertIntoPool(retset.data(), L, nn);
                    if (r < nk) nk = r;
                }
            }
        }
        if (nk <= k)
            k = nk;
        else
            ++k;
    }
    for (size_t i = 0; i < K; i++) {
        KNNs.emplace(retset[i].distance, retset[i].id);
    }
}

template<uint32_t D, uint32_t B>
void GraphRabit<D, B>::Load(const char *filename) {
    std::ifstream input(filename, std::ios::binary);
    //std::cerr << filename << std::endl;

    if (!input.is_open())
        throw std::runtime_error("Cannot open file");

    uint32_t d;
    uint32_t b;
    input.read((char *) &nd_, sizeof(uint32_t));
    input.read((char *) &d, sizeof(uint32_t));
    input.read((char *) &b, sizeof(uint32_t));
    std::cerr << d << std::endl;
    assert(d == D);
    assert(b == B);

    u = new float[B];
#if defined(RANDOM_QUERY_QUANTIZATION)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> uniform(0.0, 1.0);
    for(int i=0;i<B;i++)u[i] = uniform(gen);
#else
    for (int i = 0; i < B; i++) u[i] = 0.5;
#endif

    centroid_ = new float[B];
    data_ = new float[nd_ * D];
    binary_code = static_cast<uint64_t *>(aligned_alloc(256, nd_ * B / 64 * sizeof(uint64_t)));
    x0 = new float[nd_];
    fac = new Factor[nd_];

    input.read((char *) x0, nd_ * sizeof(float));
    input.read((char *) centroid_, B * sizeof(float));
    input.read((char *) data_, nd_ * D * sizeof(float));
    input.read((char *) binary_code, nd_ * B / 64 * sizeof(uint64_t));
    double ave_error = 0;
    for (int i = 0; i < nd_; i++) {
        float sqr_x, error, ppc, ip;
        input.read((char *) &sqr_x, sizeof(float));
        input.read((char *) &error, sizeof(float));
        input.read((char *) &ppc, sizeof(float));
        input.read((char *) &ip, sizeof(float));
        ave_error += error;
        fac[i].sqr_x = sqr_x, fac[i].error = error, fac[i].factor_ppc = ppc, fac[i].factor_ip = ip;
    }
    std::cerr << "ave error: " << ave_error / nd_ << std::endl;
    input.read((char *) &width, sizeof(unsigned));
    unsigned n_ep = 0;
    input.read((char *) &n_ep, sizeof(unsigned));
    eps_.resize(n_ep);
    input.read((char *) eps_.data(), n_ep * sizeof(unsigned));
    // width=100;
    unsigned cc = 0;
    for (int i = 0; i < nd_; i++) {
        unsigned k;
        input.read((char *) &k, sizeof(unsigned));
        cc += k;
        std::vector<unsigned> tmp(k);
        input.read((char *) tmp.data(), k * sizeof(unsigned));
        final_graph_.push_back(tmp);
    }
    cc /= nd_;
    std::cerr << "Average Degree = " << cc << std::endl;
}

template<uint32_t D, uint32_t B>
void GraphRabit<D, B>::Save(const char *filename) {
    std::ofstream output(filename, std::ios::binary);
    //std::cerr << filename << std::endl;

    output.write((char *) &nd_, sizeof(uint32_t));
    uint32_t d = D, b = B;
    output.write((char *) &d, sizeof(uint32_t));
    output.write((char *) &b, sizeof(uint32_t));

    output.write((char *) x0, nd_ * sizeof(float));
    output.write((char *) centroid_, B * sizeof(float));
    output.write((char *) data_, nd_ * D * sizeof(float));
    output.write((char *) binary_code, nd_ * B / 64 * sizeof(uint64_t));

    for (int i = 0; i < nd_; i++) {
        output.write((char *) &fac[i].sqr_x, sizeof(float));
        output.write((char *) &fac[i].error, sizeof(float));
        output.write((char *) &fac[i].factor_ppc, sizeof(float));
        output.write((char *) &fac[i].factor_ip, sizeof(float));
    }
    output.write((char *) &width, sizeof(unsigned));
    unsigned n_ep = eps_.size();
    output.write((char *) &n_ep, sizeof(unsigned));
    eps_.resize(n_ep);
    output.write((char *) eps_.data(), n_ep * sizeof(unsigned));
    // width=100;
    unsigned cc = 0;
    for (int i = 0; i < nd_; i++) {
        unsigned k = final_graph_[i].size();
        output.write((char *) &k, sizeof(unsigned));
        output.write((char *) final_graph_[i].data(), k * sizeof(unsigned));
    }
}


#endif //DEEPBIT_GRAPH_RABITQ_H
