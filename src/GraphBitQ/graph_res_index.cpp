#define EIGEN_DONT_PARALLELIZE
#define USE_AVX2
#define FAST_SCAN
//#define COUNT_SCAN

#include <iostream>
#include <fstream>

#include <ctime>
#include <cmath>
#include <matrix.h>
#include <utils.h>
#include "graph_rabitq.h"
#include "graph_res.h"
#include <getopt.h>
#include "space.h"

using namespace std;

const int MAXK = 100;

long double rotation_time = 0;
int probe_base = 50;

template<uint32_t D, uint32_t B>
void test(const Matrix<float> &Q, const Matrix<unsigned> &G, const GraphRabit<D, B> &graph, int k) {
    float sys_t, usr_t, usr_t_sum = 0, total_time = 0, search_time = 0;
    struct rusage run_start, run_end;

    // ========================================================================
    // Search Parameter

    // ========================================================================

    for (int nprobe = probe_base; nprobe <= probe_base * 20; nprobe += probe_base) {
        float total_time = 0;
        float total_ratio = 0;
        int correct = 0;

        for (int i = 0; i < Q.n; i++) {
            GetCurTime(&run_start);
            ResultHeap KNNs;
            graph.Search(Q.data + i * Q.d, k, nprobe, KNNs);
            GetCurTime(&run_end);
            GetTime(&run_start, &run_end, &usr_t, &sys_t);
            total_time += usr_t * 1e6;

            int tmp_correct = 0;
            while (KNNs.empty() == false) {
                int id = KNNs.top().second;
                KNNs.pop();
                for (int j = 0; j < k; j++)
                    if (id == G.data[i * G.d + j])tmp_correct++;
            }
            correct += tmp_correct;
//            std::cerr << "recall = " << tmp_correct << " / " << k << " " << i + 1 << " / " << Q.n << " " << usr_t * 1e6
//                      << "us" << std::endl;
        }
        float time_us_per_query = total_time / Q.n + rotation_time;
        float recall = 1.0f * correct / (Q.n * k);
        float average_ratio = total_ratio / (Q.n * k);

//        cout << "------------------------------------------------" << endl;
#ifdef COUNT_SCAN
        cout << "Count Full Scan " << count_scan << endl;
        cout << "All Distance Count " << all_dist_count << endl;
        cout << "Ratio:: " << (double) count_scan / all_dist_count << endl;
#endif
//        cout << "nprobe = " << nprobe << " k = " << k <<" Query Bits "<< B_QUERY << endl;
//        cout << "Recall = " << recall * 100.000 << "%\t" << "Ratio = " << average_ratio << endl;
//        cout << "Time = " << time_us_per_query << " us \t QPS = " << 1e6 / (time_us_per_query) << " query/s" << endl;
        cout << recall * 100.0 << " " << 1e6 / (time_us_per_query) << endl;
    }
}

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
    char result_path[256] = "";
    int subk = 0;

    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "d:r:k:s:b:", longopts, &ind);
        switch (iarg) {
            case 'k':
                if (optarg)subk = atoi(optarg);
                break;
            case 's':
                if (optarg)strcpy(source, optarg);
                break;
            case 'r':
                if (optarg)strcpy(result_path, optarg);
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
        const uint32_t BB = 128, DIM = 960;
        GraphRabit<DIM, BB> graph(X);
        graph.LoadGraph(index_path);
        probe_base = 25;
    }
    return 0;
}
