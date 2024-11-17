//
// Created by bld on 24-11-16.
//
#define EIGEN_DONT_PARALLELIZE
#define USE_AVX2

#include <iostream>
#include <cstdio>
#include <fstream>
#include <queue>
#include <getopt.h>
#include <unordered_set>
#include "utils.h"

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


    // ==============================================================================================================
    // Load Data
    char data_path[256] = "";
    char save_path[256] = "";

    sprintf(data_path, "%s%s_base.fvecs", source, dataset);
    Matrix<float> X(data_path);
    sprintf(save_path, "%s%s_base.aligned", source, dataset);
    unsigned N = X.n, D = X.d;
    unsigned Round_Up = (D * 4) % 512 == 0 ? D * 4 : ((D * 4) / 512 + 1) * 512;
    char *buffer[Round_Up];
    float *data_ptr = X.data;
    std::ofstream out(save_path, std::ios::binary);
    for (int i = 0; i < N; i++) {
        memset(buffer, 0, Round_Up);
        memcpy(buffer, data_ptr, D * 4);
        data_ptr += D;
        out.write((char*)buffer, Round_Up);
    }

    return 0;
}
