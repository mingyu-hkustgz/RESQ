#define USE_SSE
#define USE_AVX
#define USE_AVX512

#include <iostream>
#include <fstream>

#include <ctime>
#include <cmath>
#include <hnswlib/hnswlib.h>
#include "matrix.h"
#include "utils.h"
#include "space.h"
#include <getopt.h>

using namespace std;
using namespace hnswlib;

const int MAXK = 100;

long double rotation_time = 0;
int efSearch = 50;
double outer_recall = 0;

template<uint32_t D, uint32_t B>
static void get_gt(unsigned int *massQA, float *massQ, size_t vecsize, size_t qsize, L2Space &l2space,
                   size_t vecdim, vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k,
                   size_t subk, HierarchicalNSWBitQ<float, D, B> &appr_alg) {

    (vector<std::priority_queue<std::pair<float, labeltype >>>(qsize)).swap(answers);
    DISTFUNC<float> fstdistfunc_ = l2space.get_dist_func();
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < subk; j++) {
            answers[i].emplace(
                    appr_alg.fstdistfunc_(massQ + i * vecdim, appr_alg.getDataByInternalId(massQA[k * i + j]),
                                          appr_alg.dist_func_param_), massQA[k * i + j]);
        }
    }
}


int recall(std::priority_queue<std::pair<float, labeltype >> &result,
           std::priority_queue<std::pair<float, labeltype >> &gt) {
    unordered_set<labeltype> g;
    int ret = 0;
    while (gt.size()) {
        g.insert(gt.top().second);
        gt.pop();
    }
    while (result.size()) {
        if (g.find(result.top().second) != g.end()) {
            ret++;
        }
        result.pop();
    }
    return ret;
}

template<uint32_t D, uint32_t B>
static void
test_approx(float *massQ, float *randQ, size_t vecsize, size_t qsize, HierarchicalNSWBitQ<float, D, B> &appr_alg,
            size_t vecdim,
            vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k) {
    size_t correct = 0;
    size_t total = 0;
    long double total_time = 0;


    for (int i = 0; i < qsize; i++) {
#ifndef WIN32
        float sys_t, usr_t, usr_t_sum = 0;
        struct rusage run_start, run_end;
        GetCurTime(&run_start);
#endif
        std::priority_queue<std::pair<float, labeltype >> result = appr_alg.searchKnnRaBit(massQ + vecdim * i,
                                                                                           randQ + B * i, k);
#ifndef WIN32
        GetCurTime(&run_end);
        GetTime(&run_start, &run_end, &usr_t, &sys_t);
        total_time += usr_t * 1e6;
#endif
        std::priority_queue<std::pair<float, labeltype >> gt(answers[i]);
        total += gt.size();
        int tmp = recall(result, gt);
        correct += tmp;
    }
    long double time_us_per_query = total_time / qsize;
    long double recall = 1.0f * correct / total;

    cout << recall * 100.0 << " " << 1e6 / (time_us_per_query) << " " << endl;
    outer_recall = recall * 100;
    return;
}

template<uint32_t D, uint32_t B>
static void
test_vs_recall(float *massQ, float *RandQ, size_t vecsize, size_t qsize, HierarchicalNSWBitQ<float, D, B> &appr_alg,
               size_t vecdim,
               vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k) {
    vector<size_t> efs;
    unsigned efBase = efSearch;
    for (int i = 0; i < 15; i++) {
        efs.push_back(efBase);
        efBase += efSearch;
    }
    for (size_t ef: efs) {
        appr_alg.setEf(ef);
        test_approx(massQ, RandQ, vecsize, qsize, appr_alg, vecdim, answers, k);
        if (outer_recall > 99.5) break;
    }
}

int main(int argc, char *argv[]) {

    const struct option longopts[] = {
            // General Parameter
            {"help",                no_argument,       0, 'h'},

            // Query Parameter
            {"randomized",          required_argument, 0, 'd'},
            {"k",                   required_argument, 0, 'k'},
            {"epsilon0",            required_argument, 0, 'e'},
            {"gap",                 required_argument, 0, 'p'},

            // Indexing Path
            {"dataset",             required_argument, 0, 'n'},
            {"index_path",          required_argument, 0, 'i'},
            {"query_path",          required_argument, 0, 'q'},
            {"groundtruth_path",    required_argument, 0, 'g'},
            {"result_path",         required_argument, 0, 'r'},
            {"transformation_path", required_argument, 0, 't'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    char dataset[256] = "";
    char source[256] = "";
    char result_path[256] = "";
    int subk = 100, bit, D, k;

    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "k:s:d:b:r:", longopts, &ind);
        switch (iarg) {
            case 'k':
                if (optarg)subk = atoi(optarg);
                break;
            case 's':
                if (optarg)strcpy(source, optarg);
                break;
            case 'd':
                if (optarg)strcpy(dataset, optarg);
                break;
            case 'r':
                if (optarg)strcpy(result_path, optarg);
                break;
            case 'b':
                if (optarg)bit = atoi(optarg);
                break;
        }
    }

    // ================================================================================================================================
    // Data Files
    char query_path[256] = "";
    sprintf(query_path, "%s%s_query.fvecs", source, dataset);
    Matrix<float> Q(query_path);

    char groundtruth_path[256] = "";
    sprintf(groundtruth_path, "%s%s_groundtruth.ivecs", source, dataset);
    Matrix<unsigned> G(groundtruth_path);

    char transformation_path[256] = "";
    sprintf(transformation_path, "%sP.fvecs", source);
    Matrix<float> P(transformation_path);

    char index_path[256] = "";
    sprintf(index_path, "%shnsw-rabit.index", source);

    char result_file_view[256] = "";
    sprintf(result_file_view, "%s%s_hnsw_rabitq.log", result_path, dataset);

    // ================================================================================================================================

    float sys_t, usr_t, usr_t_sum = 0, total_time = 0, search_time = 0;
    struct rusage run_start, run_end;
    GetCurTime(&run_start);
    Matrix<float> RandQ(Q.n, bit, Q);
    RandQ = mul(RandQ, P);
    GetCurTime(&run_end);
    GetTime(&run_start, &run_end, &usr_t, &sys_t);
    rotation_time = usr_t * 1e6 / Q.n;
    std::string str_data(dataset);
    std::cerr << "dataset:: " << str_data << std::endl;
    vector<std::priority_queue<std::pair<float, labeltype >>> answers;
    D = Q.d;
    L2Space l2space(D);
    k = G.d;
    freopen(result_file_view, "w", stdout);
    if (str_data == "msong") {
        const uint32_t BB = 420, DIM = 420;
        auto *appr_alg = new HierarchicalNSWBitQ<float, DIM, BB>(&l2space, index_path, false);
        get_gt(G.data, Q.data, appr_alg->max_elements_, Q.n, l2space, Q.d, answers, k, subk, *appr_alg);
        test_vs_recall(Q.data, RandQ.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk);
    }
    if (str_data == "gist") {
        const uint32_t BB = 960, DIM = 960;
        auto *appr_alg = new HierarchicalNSWBitQ<float, DIM, BB>(&l2space, index_path, false);
        get_gt(G.data, Q.data, appr_alg->max_elements_, Q.n, l2space, Q.d, answers, k, subk, *appr_alg);
        test_vs_recall(Q.data, RandQ.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk);
    }
    if (str_data == "deep1M") {
        const uint32_t BB = 256, DIM = 256;
        auto *appr_alg = new HierarchicalNSWBitQ<float, DIM, BB>(&l2space, index_path, false);
        get_gt(G.data, Q.data, appr_alg->max_elements_, Q.n, l2space, Q.d, answers, k, subk, *appr_alg);
        test_vs_recall(Q.data, RandQ.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk);
    }
    if (str_data == "tiny5m") {
        const uint32_t BB = 384, DIM = 384;
        auto *appr_alg = new HierarchicalNSWBitQ<float, DIM, BB>(&l2space, index_path, false);
        get_gt(G.data, Q.data, appr_alg->max_elements_, Q.n, l2space, Q.d, answers, k, subk, *appr_alg);
        test_vs_recall(Q.data, RandQ.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk);
    }
    if (str_data == "word2vec") {
        const uint32_t BB = 320, DIM = 300;
        auto *appr_alg = new HierarchicalNSWBitQ<float, DIM, BB>(&l2space, index_path, false);
        get_gt(G.data, Q.data, appr_alg->max_elements_, Q.n, l2space, Q.d, answers, k, subk, *appr_alg);
        test_vs_recall(Q.data, RandQ.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk);
    }
    if (str_data == "glove2.2m") {
        const uint32_t BB = 320, DIM = 300;
        auto *appr_alg = new HierarchicalNSWBitQ<float, DIM, BB>(&l2space, index_path, false);
        get_gt(G.data, Q.data, appr_alg->max_elements_, Q.n, l2space, Q.d, answers, k, subk, *appr_alg);
        test_vs_recall(Q.data, RandQ.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk);
    }
    if (str_data == "OpenAI-1536") {
        const uint32_t BB = 1536, DIM = 1536;
        auto *appr_alg = new HierarchicalNSWBitQ<float, DIM, BB>(&l2space, index_path, false);
        get_gt(G.data, Q.data, appr_alg->max_elements_, Q.n, l2space, Q.d, answers, k, subk, *appr_alg);
        test_vs_recall(Q.data, RandQ.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk);
    }
    if (str_data == "OpenAI-3072") {
        const uint32_t BB = 3072, DIM = 3072;
        auto *appr_alg = new HierarchicalNSWBitQ<float, DIM, BB>(&l2space, index_path, false);
        get_gt(G.data, Q.data, appr_alg->max_elements_, Q.n, l2space, Q.d, answers, k, subk, *appr_alg);
        test_vs_recall(Q.data, RandQ.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk);
    }
    if (str_data == "msmarc-small") {
        const uint32_t BB = 1024, DIM = 1024;
        auto *appr_alg = new HierarchicalNSWBitQ<float, DIM, BB>(&l2space, index_path, false);
        get_gt(G.data, Q.data, appr_alg->max_elements_, Q.n, l2space, Q.d, answers, k, subk, *appr_alg);
        test_vs_recall(Q.data, RandQ.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk);
    }
    if (str_data == "yt1m") {
        const uint32_t BB = 1024, DIM = 1024;
        auto *appr_alg = new HierarchicalNSWBitQ<float, DIM, BB>(&l2space, index_path, false);
        get_gt(G.data, Q.data, appr_alg->max_elements_, Q.n, l2space, Q.d, answers, k, subk, *appr_alg);
        test_vs_recall(Q.data, RandQ.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk);
    }
    return 0;
}
