#define USE_AVX2
//#define COUNT_SCAN

#include <iostream>
#include <fstream>

#include <ctime>
#include <cmath>
#include <matrix.h>
#include <utils.h>
#include <ivf_rabitq.h>
#include <getopt.h>
#include "space.h"

using namespace std;

const int MAXK = 100;

long double rotation_time = 0;
int probe_base = 50;
int parallel_thread = 1;
char data_path[256] = "";

template<uint32_t D, uint32_t B>
void test(const Matrix<float> &Q, const Matrix<float> &RandQ, const Matrix<unsigned> &G,
          const IVFRN<D, B> &ivf, int k) {
    // ========================================================================
    // Search Parameter
    // ========================================================================

    for (int nprobe = probe_base; nprobe <= probe_base * 20; nprobe += probe_base) {
        int correct = 0;
        auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, parallel_thread)
        for (int i = 0; i < Q.n; i++) {
            ResultHeap KNNs = ivf.search(Q.data + i * Q.d, RandQ.data + i * RandQ.d, k, nprobe);
#pragma omp critical
            {
                int tmp_correct = 0;
                while (KNNs.empty() == false) {
                    int id = KNNs.top().second;
                    KNNs.pop();
                    for (int j = 0; j < k; j++)
                        if (id == G.data[i * G.d + j])tmp_correct++;
                }
                correct += tmp_correct;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        float total_time = duration.count() * 1e6;
        float time_us_per_query = total_time / Q.n + rotation_time;
        float recall = 1.0f * correct / (Q.n * k);
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
        iarg = getopt_long(argc, argv, "d:r:k:s:b:t:", longopts, &ind);
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
            case 't':
                if (optarg) parallel_thread = atoi(optarg);
                break;
        }
    }
    // ================================================================================================================================
    // Data Files
    char query_path[256] = "";
    sprintf(query_path, "%s%s_query.fvecs", source, dataset);
    Matrix<float> Q(query_path);

    sprintf(data_path, "%s%s_base.fvecs", source, dataset);

    char groundtruth_path[256] = "";
    sprintf(groundtruth_path, "%s%s_groundtruth.ivecs", source, dataset);
    Matrix<unsigned> G(groundtruth_path);

    char transformation_path[256] = "";
    sprintf(transformation_path, "%sP_C%d_B%d.fvecs", source, numC, bit);
    Matrix<float> P(transformation_path);

    char index_path[256] = "";
    sprintf(index_path, "%sivfrabitq%d_B%d.index", source, numC, bit);
    std::cerr << index_path << std::endl;

    char result_file_view[256] = "";
#if defined(FAST_SCAN)
    sprintf(result_file_view, "%s%s_ivf_fast_scan.log", result_path, dataset, numC, bit);
#elif defined(SCAN)
    sprintf(result_file_view, "%s%s_ivfrabitq_scan.log", result_path, dataset, numC, bit);
#endif
    std::cerr << result_file_view << std::endl;
    std::cerr << "Loading Succeed!" << std::endl;
    // ================================================================================================================================

    omp_set_num_threads(parallel_thread);
    std::cerr << "number of threads: " << parallel_thread << std::endl;
    freopen(result_file_view, "a", stdout);
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
    if (str_data == "msong") {
        const uint32_t BB = 448, DIM = 420;
        IVFRN<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 5;
        test(Q, RandQ, G, ivf, subk);
    }
    if (str_data == "gist") {
        const uint32_t BB = 960, DIM = 960;
        IVFRN<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 25;
        test(Q, RandQ, G, ivf, subk);
    }
    if (str_data == "deep1M") {
        const uint32_t BB = 256, DIM = 256;
        IVFRN<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 15;
        test(Q, RandQ, G, ivf, subk);
    }
    if (str_data == "tiny5m") {
        const uint32_t BB = 384, DIM = 384;
        IVFRN<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 25;
        test(Q, RandQ, G, ivf, subk);
    }
    if (str_data == "word2vec") {
        const uint32_t BB = 320, DIM = 300;
        IVFRN<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 15;
        test(Q, RandQ, G, ivf, subk);
    }
    if (str_data == "sift") {
        const uint32_t BB = 128, DIM = 128;
        IVFRN<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 8;
        test(Q, RandQ, G, ivf, subk);
    }
    if (str_data == "glove2.2m") {
        const uint32_t BB = 320, DIM = 300;
        IVFRN<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 15;
        test(Q, RandQ, G, ivf, subk);
    }
    if (str_data == "OpenAI-1536") {
        const uint32_t BB = 1536, DIM = 1536;
        IVFRN<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 30;
        test(Q, RandQ, G, ivf, subk);
    }
    if (str_data == "OpenAI-3072") {
        const uint32_t BB = 3072, DIM = 3072;
        IVFRN<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 30;
        test(Q, RandQ, G, ivf, subk);
    }
    if (str_data == "msmarc10m") {
        const uint32_t BB = 1024, DIM = 1024;
        IVFRN<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 30;
        test(Q, RandQ, G, ivf, subk);
    }
    if (str_data == "yt1m") {
        const uint32_t BB = 1024, DIM = 1024;
        IVFRN<DIM, BB> ivf;
        ivf.load(index_path);
        probe_base = 30;
        test(Q, RandQ, G, ivf, subk);
    }
    return 0;
}
