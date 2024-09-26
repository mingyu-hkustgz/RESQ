//
// Created by BLD on 24-9-23.
//
#include <vector>
#include "utils.h"

#ifndef DEEPBIT_GRAPH_H
#define DEEPBIT_GRAPH_H

struct Neighbor {
    unsigned id;
    float distance;
    bool flag;

    Neighbor() = default;
    Neighbor(unsigned id, float distance, bool f)
            : id{id}, distance{distance}, flag(f) {}

    inline bool operator<(const Neighbor &other) const {
        return distance < other.distance;
    }
};

struct SimpleNeighbor {
    unsigned id;
    float distance;

    SimpleNeighbor() = default;
    SimpleNeighbor(unsigned id_, float distance_)
            : id(id_), distance(distance_) {}

    inline bool operator<(const SimpleNeighbor &other) const {
        return distance < other.distance;
    }
};

struct SimpleNeighbors {
    std::vector<SimpleNeighbor> pool;
};
static inline int InsertIntoPool(Neighbor *addr, unsigned K, Neighbor nn) {
    // find the location to insert
    int left = 0, right = K - 1;
    if (addr[left].distance > nn.distance) {
        memmove((char *)&addr[left + 1], &addr[left], K * sizeof(Neighbor));
        addr[left] = nn;
        return left;
    }
    if (addr[right].distance < nn.distance) {
        addr[K] = nn;
        return K;
    }
    while (left < right - 1) {
        int mid = (left + right) / 2;
        if (addr[mid].distance > nn.distance)
            right = mid;
        else
            left = mid;
    }
    // check equal ID

    while (left > 0) {
        if (addr[left].distance < nn.distance) break;
        if (addr[left].id == nn.id) return K + 1;
        left--;
    }
    if (addr[left].id == nn.id || addr[right].id == nn.id) return K + 1;
    memmove((char *)&addr[right + 1], &addr[right],
            (K - right) * sizeof(Neighbor));
    addr[right] = nn;
    return right;
}

void GenRandom(std::mt19937& rng, unsigned* addr, unsigned size, unsigned N) {
    for (unsigned i = 0; i < size; ++i) {
        addr[i] = rng() % (N - size);
    }
    std::sort(addr, addr + size);
    for (unsigned i = 1; i < size; ++i) {
        if (addr[i] <= addr[i - 1]) {
            addr[i] = addr[i - 1] + 1;
        }
    }
    unsigned off = rng() % N;
    for (unsigned i = 0; i < size; ++i) {
        addr[i] = (addr[i] + off) % N;
    }
}


#endif //DEEPBIT_GRAPH_H
