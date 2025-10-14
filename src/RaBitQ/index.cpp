#define EIGEN_DONT_PARALLELIZE
#define USE_AVX2

#include <iostream>
#include <cstdio>
#include <fstream>
#include <queue>
#include <getopt.h>
#include <unordered_set>

#include "matrix.h"
#include "utils.h"
#include "ivf_rabitq.h"

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
        iarg = getopt_long(argc, argv, "d:s:b:c:", longopts, &ind);
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
            case 'c':
                if (optarg) numC = atoi(optarg);
                break;
        }
    }


    // ==============================================================================================================
    // Load Data
    char data_path[256] = "";
    char index_path[256] = "";
    char centroid_path[256] = "";
    char x0_path[256] = "";
    char dist_to_centroid_path[256] = "";
    char cluster_id_path[256] = "";
    char binary_path[256] = "";

    sprintf(data_path, "%s%s_base.fvecs", source, dataset);
    Matrix<float> X(data_path);

    sprintf(centroid_path, "%sRandCentroid_C%d_B%d.fvecs", source, numC, bit);
    Matrix<float> C(centroid_path);

    sprintf(x0_path, "%sx0_C%d_B%d.fvecs", source, numC, bit);
    Matrix<float> x0(x0_path);

    sprintf(dist_to_centroid_path, "%s%s_dist_to_centroid_%d.fvecs", source, dataset, numC);
    Matrix<float> dist_to_centroid(dist_to_centroid_path);

    sprintf(cluster_id_path, "%s%s_cluster_id_%d.ivecs", source, dataset, numC);
    Matrix <uint64_t> cluster_id(cluster_id_path);

    sprintf(binary_path, "%sRandNet_C%d_B%d.Ivecs", source, numC, bit);
    Matrix <uint64_t> binary(binary_path);

    sprintf(index_path, "%sivfrabitq%d_B%d.index", source, numC, bit);
    std::cerr << "Loading Succeed!" << std::endl << std::endl;
    // ==============================================================================================================
    std::string str_data(dataset);
    std::cerr << "dataset:: " << str_data << std::endl;
    if (str_data == "msong") {
        const uint64_t BB = 448, DIM = 420;
        IVFRN <DIM, BB> ivf(X, C, dist_to_centroid, x0, cluster_id, binary);
        ivf.save(index_path);
    }
    if (str_data == "gist") {
        const uint64_t BB = 960, DIM = 960;
        IVFRN <DIM, BB> ivf(X, C, dist_to_centroid, x0, cluster_id, binary);
        ivf.save(index_path);
    }
    if (str_data == "deep1M") {
        const uint64_t BB = 256, DIM = 256;
        IVFRN <DIM, BB> ivf(X, C, dist_to_centroid, x0, cluster_id, binary);
        ivf.save(index_path);
    }
    if (str_data == "tiny5m") {
        const uint64_t BB = 384, DIM = 384;
        IVFRN <DIM, BB> ivf(X, C, dist_to_centroid, x0, cluster_id, binary);
        ivf.save(index_path);
    }
    if (str_data == "word2vec") {
        const uint64_t BB = 320, DIM = 300;
        IVFRN <DIM, BB> ivf(X, C, dist_to_centroid, x0, cluster_id, binary);
        ivf.save(index_path);
    }
    if (str_data == "sift") {
        const uint64_t BB = 128, DIM = 128;
        IVFRN <DIM, BB> ivf(X, C, dist_to_centroid, x0, cluster_id, binary);
        ivf.save(index_path);
    }
    if (str_data == "glove2.2m") {
        const uint64_t BB = 320, DIM = 300;
        IVFRN <DIM, BB> ivf(X, C, dist_to_centroid, x0, cluster_id, binary);
        ivf.save(index_path);
    }
    if (str_data == "OpenAI-1536") {
        const uint64_t BB = 1536, DIM = 1536;
        IVFRN <DIM, BB> ivf(X, C, dist_to_centroid, x0, cluster_id, binary);
        ivf.save(index_path);
    }
    if (str_data == "OpenAI-3072") {
        const uint64_t BB = 3072, DIM = 3072;
        IVFRN <DIM, BB> ivf(X, C, dist_to_centroid, x0, cluster_id, binary);
        ivf.save(index_path);
    }
    if (str_data.find("msmarc") != std::string::npos) {
        const uint64_t BB = 1024, DIM = 1024;
        auto start_time = std::chrono::high_resolution_clock::now();
        IVFRN<DIM, BB> ivf(X, C, dist_to_centroid, x0, cluster_id, binary);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        std::ofstream out("./results/time-log/"+str_data+"/Rabit-Index-Time.log", std::ios::app);
        out<<"Rabit Naive Index Time:"<<duration.count()<<" (s)";
        ivf.save(index_path);
    }
    if (str_data == "yt1m") {
        const uint64_t BB = 1024, DIM = 1024;
        IVFRN <DIM, BB> ivf(X, C, dist_to_centroid, x0, cluster_id, binary);
        ivf.save(index_path);
    }
    return 0;
}
