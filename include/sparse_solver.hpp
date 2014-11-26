#ifndef _SPARSE_SOLVE_HPP_
#define _SPARSE_SOLVE_HPP_

#include <vector>
#include <map>
#include <Eigen/Core>
#include <types.hpp>

namespace ssolver {
    // standard regression functions
    VectorT ridge(const MatrixT &X, const VectorT &y, g_float lambd2);
    
    // sparse regression implementations
    VectorT group_fista(MatrixT &X, const VectorT &y, const struct params &p, const int N, group_stack &stack);
    VectorT fista(MatrixT &X, const VectorT &y, const struct params &p, const int N);
}

#endif
