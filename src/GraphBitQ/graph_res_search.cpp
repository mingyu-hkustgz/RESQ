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
#include "graph_res.h"
#include <getopt.h>
#include "space.h"

using namespace std;

const int MAXK = 100;

long double rotation_time = 0;
int probe_base = 50;

template<uint32_t D, uint32_t B>
void test(const Matrix<float> &Q,const Matrix<float> &RandQ, const Matrix<unsigned> &G, const GraphRes<D, B> &graph, int k) {
    float sys_t, usr_t, usr_t_sum = 0, total_time = 0, search_time = 0;
    struct rusage run_start, run_end;

    // ========================================================================
    // Search Parameter

    // ========================================================================

    for (int nprobe = probe_base; nprobe <= probe_base * 20; nprobe += probe_base) {
        if (nprobe < k) continue;
        float total_time = 0;
        float total_ratio = 0;
        int correct = 0;
#ifdef COUNT_SCAN
        count_scan = 0;
        all_dist_count = 0;
#endif
        for (int i = 0; i < Q.n; i++) {
            GetCurTime(&run_start);
            ResultHeap KNNs;
            graph.SearchRes(Q.data + i * Q.d, RandQ.data + i * RandQ.d, k, nprobe, KNNs);
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
    char query_path[256] = "";
    sprintf(query_path, "%s%s_query.fvecs", source, dataset);
    Matrix<float> Q(query_path);

    char data_path[256] = "";
    sprintf(data_path, "%s%s_proj.fvecs", source, dataset);
    Matrix<float> X(data_path);

    char groundtruth_path[256] = "";
    sprintf(groundtruth_path, "%s%s_groundtruth.ivecs", source, dataset);
    Matrix<unsigned> G(groundtruth_path);

    char transformation_path[256] = "";
    sprintf(transformation_path, "%sRESP.fvecs", source);
    Matrix<float> P(transformation_path);

    char PCA_matrix_path[256] = "";
    sprintf(PCA_matrix_path, "%s%s_pca.fvecs", source, dataset);
    Matrix<float> PCA(PCA_matrix_path);

    char index_path[256] = "";
    sprintf(index_path, "%s%s_res.graph", source, dataset);
    std::cerr << index_path << std::endl;

    // ================================================================================================================================
    float sys_t, usr_t, usr_t_sum = 0, total_time = 0, search_time = 0;
    struct rusage run_start, run_end;
    GetCurTime(&run_start);
    char result_file_view[256] = "";
    sprintf(result_file_view, "%s%s_graph_res.log", result_path, dataset);
    std::cerr << "begin Matrix Operation" << std::endl;
    Matrix<float> PCAQ(Q.n, Q.d, Q);
    PCAQ = mul(PCAQ, PCA);
    auto TEMP_Q = resize_matrix(PCAQ, PCAQ.n, bit);
    auto RandQ = mul(TEMP_Q, P);
    GetCurTime(&run_end);
    GetTime(&run_start, &run_end, &usr_t, &sys_t);
    rotation_time = usr_t * 1e6 / Q.n;
    freopen(result_file_view, "a", stdout);
    std::string str_data(dataset);
    if (str_data == "gist") {
        const uint32_t BB = 256, DIM = 960;
        GraphRes<DIM, BB> graph;
        graph.Load(index_path);
        probe_base = 25;
        test(PCAQ, RandQ, G, graph, subk);
    }
    if (str_data == "deep1M") {
        const uint32_t BB = 128, DIM = 256;
        GraphRes<DIM, BB> graph;
        graph.Load(index_path);
        probe_base = 25;
        test(PCAQ, RandQ, G, graph, subk);
    }
    if (str_data == "sift") {
        const uint32_t BB = 64, DIM = 128;
        GraphRes<DIM, BB> graph;
        graph.Load(index_path);
        probe_base = 25;
        test(PCAQ, RandQ, G, graph, subk);
    }
    if (str_data == "msong") {
        const uint32_t BB = 128, DIM = 420;
        GraphRes<DIM, BB> graph;
        graph.Load(index_path);
        probe_base = 20;
        test(PCAQ, RandQ, G, graph, subk);
    }
    if (str_data == "tiny5m") {
        const uint32_t BB = 128, DIM = 384;
        GraphRes<DIM, BB> graph;
        graph.Load(index_path);
        probe_base = 100;
        test(PCAQ, RandQ, G, graph, subk);
    }
    if (str_data == "glove2.2m") {
        const uint32_t BB = 256, DIM = 300;
        GraphRes<DIM, BB> graph;
        graph.Load(index_path);
        probe_base = 100;
        test(PCAQ, RandQ, G, graph, subk);
    }
    if (str_data == "word2vec") {
        const uint32_t BB = 256, DIM = 300;
        GraphRes<DIM, BB> graph;
        graph.Load(index_path);
        probe_base = 100;
        test(PCAQ, RandQ, G, graph, subk);
    }
    return 0;
}
