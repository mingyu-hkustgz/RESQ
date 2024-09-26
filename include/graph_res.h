//
// Created by BLD on 24-9-23.
//

#ifndef DEEPBIT_GRAPH_RES_H
#define DEEPBIT_GRAPH_RES_H

#include "fast_scan.h"
#include "space.h"
#include "utils.h"
#include "matrix.h"
#include "graph.h"
#include <boost/dynamic_bitset.hpp>

class GraphRes {
private:
public:
    GraphRes();

    void Search(const float *query, const float *x, int k,
                        const int L, unsigned *indices);

    void Save(const char *filename);

    void Load(const char *filename);


};

#endif //DEEPBIT_GRAPH_RES_H
