#include <iostream>
#include <fstream>
#include <queue>
#include <getopt.h>
#include <unordered_set>

#include "matrix.h"
#include "utils.h"
#include "hnswlib/hnswlib.h"

using namespace std;
using namespace hnswlib;

template<uint32_t D, uint32_t B>
void index_build(const Matrix<float> *X, HierarchicalNSWBitQ<float, D, B> *hnsw) {
    auto N = X->n;
    unsigned check_tag = 1, report = 100000;
#pragma omp parallel for schedule(dynamic, 144)
    for (int i = 1; i < N; i++) {
        hnsw->addPoint(X->data + i * D, i);
#pragma omp critical
        {
            check_tag++;
            if (check_tag % report == 0) {
                cerr << "Processing - " << check_tag << " / " << N << endl;
            }
        }
    }
}


int main(int argc, char *argv[]) {

    const struct option longopts[] = {
            // General Parameter
            {"help",           no_argument,       0, 'h'},

            // Index Parameter
            {"efConstruction", required_argument, 0, 'e'},
            {"M",              required_argument, 0, 'm'},

            // Indexing Path
            {"data_path",      required_argument, 0, 'd'},
            {"index_path",     required_argument, 0, 'i'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    char dataset[256] = "";
    char source[256] = "";
    char data_path[256] = "";
    size_t efConstruction = 1000;
    size_t M = 32;

    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "d:s:", longopts, &ind);
        switch (iarg) {
            case 'd':
                if (optarg) {
                    strcpy(dataset, optarg);
                }
                break;
            case 's':
                if (optarg) {
                    strcpy(source, optarg);
                }
                break;
        }
    }

    sprintf(data_path, "%s%s_proj.fvecs", source, dataset);
    Matrix<float> *X = new Matrix<float>(data_path);

    char x0_path[256] = "";
    sprintf(x0_path, "%sRES_x0.fvecs", source);
    Matrix<float> x0(x0_path);

    char dist_to_centroid_path[256] = "";
    sprintf(dist_to_centroid_path, "%sRES_x2.fvecs", source);
    Matrix<float> x2(dist_to_centroid_path);

    char centroid_path[256] = "";
    sprintf(centroid_path, "%sRESCentroid.fvecs", source);
    Matrix<float> C(centroid_path);

    char binary_path[256] = "";
    sprintf(binary_path, "%sRES_Rand.Ivecs", source);
    Matrix<uint64_t> binary(binary_path);

    char mean_path[256] = "";
    sprintf(mean_path, "%s%s_mean.fvecs", source, dataset);
    Matrix<float> Mean(mean_path);

    char index_path[256] = "";
    sprintf(index_path, "%shnsw-res.index", source);

    size_t D = X->d;
    size_t N = X->n;
    size_t report = 50000;
    std::string str_data(dataset);
    std::cerr << "dataset:: " << str_data << std::endl;
    L2Space l2space(D);
    if (str_data == "msong") {
        const uint32_t BB = 128, DIM = 420;
        HierarchicalNSWBitQ<float, DIM, BB> *appr_alg = new HierarchicalNSWBitQ<float, DIM, BB>(&l2space, N, M,
                                                                                                efConstruction);
        appr_alg->addPoint(X->data, 0);
        index_build(X, appr_alg);
        appr_alg->res_quantized_init(C, x2, x0, binary, Mean);
        appr_alg->saveIndex(index_path);
    }
    if (str_data == "gist") {
        const uint32_t BB = 128, DIM = 960;
        HierarchicalNSWBitQ<float, DIM, BB> *appr_alg = new HierarchicalNSWBitQ<float, DIM, BB>(&l2space, N, M,
                                                                                                efConstruction);
        appr_alg->addPoint(X->data, 0);
        index_build(X, appr_alg);
        appr_alg->res_quantized_init(C, x2, x0, binary, Mean);
        appr_alg->saveIndex(index_path);
    }
    if (str_data == "deep1M") {
        const uint32_t BB = 128, DIM = 256;
        HierarchicalNSWBitQ<float, DIM, BB> *appr_alg = new HierarchicalNSWBitQ<float, DIM, BB>(&l2space, N, M,
                                                                                                efConstruction);
        appr_alg->addPoint(X->data, 0);
        index_build(X, appr_alg);
        appr_alg->res_quantized_init(C, x2, x0, binary, Mean);
        appr_alg->saveIndex(index_path);
    }
    if (str_data == "tiny5m") {
        const uint32_t BB = 128, DIM = 384;
        HierarchicalNSWBitQ<float, DIM, BB> *appr_alg = new HierarchicalNSWBitQ<float, DIM, BB>(&l2space, N, M,
                                                                                                efConstruction);
        appr_alg->addPoint(X->data, 0);
        index_build(X, appr_alg);
        appr_alg->res_quantized_init(C, x2, x0, binary, Mean);
        appr_alg->saveIndex(index_path);
    }
    if (str_data == "word2vec") {
        const uint32_t BB = 256, DIM = 300;
        HierarchicalNSWBitQ<float, DIM, BB> *appr_alg = new HierarchicalNSWBitQ<float, DIM, BB>(&l2space, N, M,
                                                                                                efConstruction);
        appr_alg->addPoint(X->data, 0);
        index_build(X, appr_alg);
        appr_alg->res_quantized_init(C, x2, x0, binary, Mean);
        appr_alg->saveIndex(index_path);
    }
    if (str_data == "sift") {
        const uint32_t BB = 64, DIM = 128;
        HierarchicalNSWBitQ<float, DIM, BB> *appr_alg = new HierarchicalNSWBitQ<float, DIM, BB>(&l2space, N, M,
                                                                                                efConstruction);
        appr_alg->addPoint(X->data, 0);
        index_build(X, appr_alg);
        appr_alg->res_quantized_init(C, x2, x0, binary, Mean);
        appr_alg->saveIndex(index_path);
    }
    if (str_data == "glove2.2m") {
        const uint32_t BB = 256, DIM = 300;
        HierarchicalNSWBitQ<float, DIM, BB> *appr_alg = new HierarchicalNSWBitQ<float, DIM, BB>(&l2space, N, M,
                                                                                                efConstruction);
        appr_alg->addPoint(X->data, 0);
        index_build(X, appr_alg);
        appr_alg->res_quantized_init(C, x2, x0, binary, Mean);
        appr_alg->saveIndex(index_path);
    }
    if (str_data == "OpenAI-1536") {
        const uint32_t BB = 512, DIM = 1536;
        HierarchicalNSWBitQ<float, DIM, BB> *appr_alg = new HierarchicalNSWBitQ<float, DIM, BB>(&l2space, N, M,
                                                                                                efConstruction);
        appr_alg->addPoint(X->data, 0);
        index_build(X, appr_alg);
        appr_alg->res_quantized_init(C, x2, x0, binary, Mean);
        appr_alg->saveIndex(index_path);
    }
    if (str_data == "OpenAI-3072") {
        const uint32_t BB = 512, DIM = 3072;
        HierarchicalNSWBitQ<float, DIM, BB> *appr_alg = new HierarchicalNSWBitQ<float, DIM, BB>(&l2space, N, M,
                                                                                                efConstruction);
        appr_alg->addPoint(X->data, 0);
        index_build(X, appr_alg);
        appr_alg->res_quantized_init(C, x2, x0, binary, Mean);
        appr_alg->saveIndex(index_path);
    }
    if (str_data == "msmarc-small") {
        const uint32_t BB = 512, DIM = 1024;
        HierarchicalNSWBitQ<float, DIM, BB> *appr_alg = new HierarchicalNSWBitQ<float, DIM, BB>(&l2space, N, M,
                                                                                                efConstruction);
        appr_alg->addPoint(X->data, 0);
        index_build(X, appr_alg);
        appr_alg->res_quantized_init(C, x2, x0, binary, Mean);
        appr_alg->saveIndex(index_path);
    }
    if (str_data == "yt1m") {
        const uint32_t BB = 512, DIM = 1024;
        HierarchicalNSWBitQ<float, DIM, BB> *appr_alg = new HierarchicalNSWBitQ<float, DIM, BB>(&l2space, N, M,
                                                                                                efConstruction);
        appr_alg->addPoint(X->data, 0);
        index_build(X, appr_alg);
        appr_alg->res_quantized_init(C, x2, x0, binary, Mean);
        appr_alg->saveIndex(index_path);
    }
    return 0;
}
