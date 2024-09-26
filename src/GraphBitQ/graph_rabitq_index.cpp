#define EIGEN_DONT_PARALLELIZE
#define USE_AVX2
#include <iostream>
#include <fstream>

#include <ctime>
#include <cmath>
#include <matrix.h>
#include <utils.h>
#include "graph_rabitq.h"
#include <getopt.h>

using namespace std;

int main(int argc, char *argv[]) {

    const struct option longopts[] = {
            // General Parameter
            {"help",        no_argument,       0, 'h'},

            // Query Parameter
            {"K",           required_argument, 0, 'k'},

            // Indexing Path
            {"dataset",     required_argument, 0, 'd'},
            {"source",      required_argument, 0, 's'},
            {"result_path", required_argument, 0, 'r'},
    };

    int ind, bit;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    char dataset[256] = "";
    char source[256] = "";

    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "s:d:b:", longopts, &ind);
        switch (iarg) {
            case 's':
                if (optarg)strcpy(source, optarg);
                break;
            case 'd':
                if (optarg)strcpy(dataset, optarg);
                break;
            case 'b':
                if (optarg) bit = atoi(optarg);
                break;
        }
    }

    // ================================================================================================================================
    // Data Files
    char data_path[256] = "";
    sprintf(data_path, "%s%s_base.fvecs", source, dataset);
    Matrix<float> X(data_path);

    char transformation_path[256] = "";
    sprintf(transformation_path, "%sP.fvecs", source);
    Matrix<float> P(transformation_path);

    char x0_path[256] = "";
    sprintf(x0_path, "%sx0.fvecs", source);
    Matrix<float> x0(x0_path);

    char dist_to_centroid_path[256] = "";
    sprintf(dist_to_centroid_path, "%sx2.fvecs", source);
    Matrix<float> x2(dist_to_centroid_path);

    char centroid_path[256] = "";
    sprintf(centroid_path, "%sRandCentroid.fvecs", source);
    Matrix<float> C(centroid_path);

    char binary_path[256] = "";
    sprintf(binary_path, "%sRandNet.Ivecs", source);
    Matrix<uint64_t> binary(binary_path);

    char graph_path[256] = "";
    sprintf(graph_path, "%s%s.ssg", source, dataset);

    char index_path[256] = "";
    sprintf(index_path, "%s%s_rabit.graph", source, dataset);
    std::cerr << graph_path << std::endl;

    // ================================================================================================================================
    std::string str_data(dataset);
    std::cerr << "dataset:: " << str_data << std::endl;
    if (str_data == "gist") {
        const uint32_t BB = 960, DIM = 960;
        GraphRabit<DIM, BB> graph(X, C, x2, x0, binary);
        std::cerr << "load binary finished" << std::endl;
        graph.LoadGraph(graph_path);
        std::cerr<<"begin test"<<std::endl;
        graph.Save(index_path);
    }
    return 0;
}
