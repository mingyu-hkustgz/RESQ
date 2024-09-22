#define USE_AVX2
#define FAST_SCAN
#include <iostream>
#include <cstdio>
#include <fstream>
#include <queue>
#include <getopt.h>
#include <unordered_set>

#include "matrix.h"
#include "utils.h"
#include "ivf_res.h"
#include "omp.h"

using namespace std;

int main(int argc, char *argv[]) {

    const struct option longopts[] = {
            // General Parameter
            {"help",    no_argument,       0, 'h'},

            // Indexing Path
            {"dataset", required_argument, 0, 'd'},
            {"source",  required_argument, 0, 's'},
    };

    int ind, bit;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    char dataset[256] = "";
    char source[256] = "";
    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "d:s:b:", longopts, &ind);
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
            case 'b':
                if (optarg) bit = atoi(optarg);
                break;
        }
    }


    // ==============================================================================================================
    // Load Data
    char data_path[256] = "";
    char index_path[256] = "";
    char centroid_path[256] = "";
    char x0_path[256] = "";
    char cluster_id_path[256] = "";
    char binary_path[256] = "";
    char dist_to_centroid_path[256] = "";

    sprintf(data_path, "%s%s_proj.fvecs", source, dataset);
    Matrix<float> X(data_path);

    sprintf(centroid_path, "%sRESCentroid_C%d_B%d.fvecs", source, numC, bit);
    Matrix<float> C(centroid_path);

    sprintf(x0_path, "%sRES_x0_C%d_B%d.fvecs", source, numC, bit);
    Matrix<float> x0(x0_path);

    sprintf(dist_to_centroid_path, "%sp%s_dist_to_centroid_%d.fvecs", source, dataset, numC);
    Matrix<float> dist_to_centroid(dist_to_centroid_path);

    sprintf(cluster_id_path, "%sp%s_cluster_id_%d.ivecs", source, dataset, numC);
    Matrix<uint32_t> cluster_id(cluster_id_path);

    sprintf(binary_path, "%sRES_Rand_C%d_B%d.Ivecs", source, numC, bit);
    Matrix<uint64_t> binary(binary_path);

#ifdef RESIDUAL_SPLIT
    sprintf(index_path, "%sivf_split%d_B%d.index", source, numC, bit);
#else
    sprintf(index_path, "%sivf_res%d_B%d.index", source, numC, bit);
#endif
    std::cerr << "Loading Succeed!" << std::endl << std::endl;
    // ==============================================================================================================
    std::string str_data(dataset);
    std::cerr << "dataset:: " << str_data << std::endl;
    if (str_data == "msong") {
        const uint32_t BB = 128, DIM = 420;
        IVFRES<DIM, BB> ivf(X, C, dist_to_centroid, x0, cluster_id, binary);
        ivf.save(index_path);
    }
    if (str_data == "gist") {
        const uint32_t BB = 128, DIM = 960;
        IVFRES<DIM, BB> ivf(X, C, dist_to_centroid, x0, cluster_id, binary);
        ivf.save(index_path);
    }
    if (str_data == "deep1M") {
        const uint32_t BB = 128, DIM = 256;
        IVFRES<DIM, BB> ivf(X, C, dist_to_centroid, x0, cluster_id, binary);
        ivf.save(index_path);
    }
    if (str_data == "tiny5m") {
        const uint32_t BB = 128, DIM = 384;
        IVFRES<DIM, BB> ivf(X, C, dist_to_centroid, x0, cluster_id, binary);
        ivf.save(index_path);
    }
    if (str_data == "word2vec") {
        const uint32_t BB = 256, DIM = 300;
        IVFRES<DIM, BB> ivf(X, C, dist_to_centroid, x0, cluster_id, binary);
        ivf.save(index_path);
    }
    if (str_data == "glove2.2m") {
        const uint32_t BB = 256, DIM = 300;
        IVFRES<DIM, BB> ivf(X, C, dist_to_centroid, x0, cluster_id, binary);
        ivf.save(index_path);
    }

    return 0;
}
