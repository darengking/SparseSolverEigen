#ifndef _SPARSE_SOLVE_TYPES_H_
#define _SPARSE_SOLVE_TYPES_H_

#include <vector>
#include <Eigen/Core>

namespace ssolver {
    /* define basic types  */
    typedef double g_float;
    typedef Eigen::MatrixXd MatrixT;
    typedef Eigen::VectorXd VectorT;
    typedef std::pair<int,int> group;
    typedef std::vector<std::pair<group, g_float> > group_stack;
    
    struct params {
        g_float lambd1, lambdm, lambd2, lambd2d, tol;
        g_float L;
        bool group_sparse;
        bool refit;
        int it_tol;
        VectorT w0;
        
        params() {
                lambd1 = 1e-7;
                lambdm = 1e-4;
                L = 1;
                tol = 1e-5; 
                lambd2 = 1e-4;
                lambd2d = 1e-4;
                refit = false;
                it_tol = 10;
                w0 = VectorT();
        }
    };
    
    
    /* basic helpers that to not fit in any other file */
    inline g_float SQR(g_float x) {
        return x * x;
    }
}
    
#endif
