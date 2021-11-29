#pragma once

#include "DPHPC.h"
#include <vector>

NAMESPACE_DPHPC_BEGIN

template<typename T> void applyPermutation(std::vector<T>& v, unsigned int indices[], unsigned int n) {
    
    using std::swap; // to permit Koenig lookup
    for (size_t i = 0; i < n; i++) {
        auto current = i;
        while (i != indices[current]) {
            auto next = indices[current];
            swap(v[current], v[next]);
            indices[current] = current;
            current = next;
        }
    indices[current] = current;
 }
}

NAMESPACE_DPHPC_END