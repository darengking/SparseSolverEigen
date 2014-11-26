#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <log_macros.h>
#include <sparse_solver.hpp>

namespace ssolver {
    
    /* little helpers */
    inline g_float GROUP_VALUE(const group &curr_group, const VectorT &a, const VectorT &beta, const VectorT &sorted_ind, int d, const struct params &p) {
        g_float curr_v = 0.;
        for (int i = curr_group.first; i <= curr_group.second; ++i) {
            curr_v += a(sorted_ind[i]) - (2. / p.L * (p.lambd1 + p.lambdm * (d - i + 1)));
        }
        curr_v /= 2. * (curr_group.second - curr_group.first + 1);
        curr_v = std::max(curr_v, 0.);
        return curr_v;
    }
    
    inline group GROUP_COMBINE(const group &g1, const group &g2) {
        return std::make_pair(std::min(g1.first, g2.first), std::max(g1.second, g2.second));
    }
    
    void sort(const VectorT &a, VectorT &indices) {
        // perform a simple insertion sort
        int n = indices.size();
        for (int i = 1; i < n; ++i) {
                g_float tmp_v = a(indices(i));
                int hole = i;
                // find the correct spot for this entry
                while (hole > 0 && a(indices(hole-1)) < tmp_v) {
                    // move this element one to the right
                    indices[hole] = indices[hole - 1];
                    --hole;
                }
                // hole should now be the spot for our new entry
                indices[hole] = i;
        }
    }
    
    /* helpers for fista methods */
    
    VectorT oscar_project(const VectorT &pred_err, const VectorT &deriv, const VectorT &beta, 
                          struct params &p, group_stack &stack, VectorT &sorted_ind) {
        int d = beta.size();
        // first calculate all the a values
        VectorT a(d);
        for (int i = 0; i < d; ++i) {
            a(i) = 2. * std::fabs(p.L * beta(i) - deriv(i)) / p.L;
            // std::cout << "a " << a(i) << " b " <<  beta(i) << " db " << deriv(i) << std::endl;
        }
        // next sort them and retrieve the sorted indices
        sorted_ind.resize(d);
        for (int  i = 0; i < d; ++i)
            sorted_ind(i) = i;
        sort(a, sorted_ind);
        /*
          for (int i = 0; i < d; ++i) {
          std::cout << sorted_ind(i) << ":" <<  a(sorted_ind(i)) << " ";
          }
          std::cout << std::endl;
        */
        // push the first group (denoted by the first index to the stack)
        group group1 = std::make_pair(0,0);
        stack.push_back(std::make_pair(group1, GROUP_VALUE(group1, a, beta, sorted_ind, d, p)));
        for (int i = 1; i < d; ++i) {
            group curr_group = std::make_pair(i,i);
            g_float curr_v = GROUP_VALUE(curr_group, a, beta, sorted_ind, d, p);
            while (!stack.empty() && curr_v >= stack.back().second) {
                group &n_group = stack.back().first;
                LOG(5, << "combining groups (" << n_group.first << "," << n_group.second << ") (" << curr_group.first << "," << curr_group.second << ")");
                curr_group = GROUP_COMBINE(curr_group, n_group);
                curr_v = GROUP_VALUE(curr_group, a, beta, sorted_ind, d, p);
                stack.pop_back();
            }
            stack.push_back(std::make_pair(curr_group, curr_v));
        }
        // at this point we have discovered the optimal group structure
        /*  finally calculate beta_star */
        // first recover the magnitude as the group value
        VectorT beta_star(d);
        int tmp_i = 0;
        for (group_stack::const_iterator it = stack.begin(); it != stack.end(); ++it) {
            const group &group = it->first;
            for (int j = group.first; j <= group.second; ++j) {
                beta_star[sorted_ind(j)] = it->second;
            }
            // DEBUG ONLY
            if (DEBUG >= 2) {
                if (group.second - group.first > 0) {
                    std::cout << "Group " << tmp_i << " (" << group.first << "," << group.second << ") =";
                    for (int j = group.first; j <= group.second; ++j) {
                        std::cout << " " << sorted_ind(j);
                    }
                    std::cout << std::endl;
                }
            }
            ++tmp_i;
            // END DEBUG ONLY
        }
        // next recover the sign
        for (int i = 0; i < d; ++i) {
            if ((p.L * beta(i) - deriv(i)) * beta_star(i) < 0) {
                // revert the sign
                beta_star(i) *= -1.;
            }
        }
        return beta_star;
    }
    
    
    VectorT l1_project(const VectorT &pred_err, const VectorT &deriv, const VectorT &beta, struct params &p) {
        VectorT res = beta - (1. / p.L) * deriv;
        // shrink towards zero and threshold
        for (int i = 0; i < res.size(); ++i) {
            if (res(i) > p.lambd1 / p.L) {
                res(i) -= p.lambd1;
            } else if (res(i) < -p.lambd1 / p.L) {
                res(i) += p.lambd1;
            } else {
                res(i) = 0.;
            }
        } 
        return res;
    }
    
    VectorT projection_step(const MatrixT &X, const VectorT &y, const VectorT &beta, struct params &p, group_stack &stack, VectorT &sorted_ind) {
        // calculate the gradient
        VectorT pred_err = (y - X * beta);
        VectorT deriv = (-X.transpose() * pred_err);
        bool stop_backtrack = false;
        VectorT res;
        res.resize(beta.size());
        int iter = 1;
        int max_iter_backtracking = 50;
        while (!stop_backtrack && iter < max_iter_backtracking) {
            // clear the groups before computing the projection
            stack.clear();
            if (p.group_sparse)
                res = oscar_project(pred_err, deriv, beta, p, stack, sorted_ind);
            else 
                res = l1_project(pred_err, deriv, beta, p);
            g_float pred_error_new = 0.5 * (y - X * res).squaredNorm();
            g_float tmp = 0.5 * pred_err.squaredNorm() + (res - beta).transpose().dot(deriv) + (p.L/2.) * (res-beta).squaredNorm();
            if (pred_error_new <= tmp) {
                stop_backtrack = true;
            } else {
                // update L
                p.L = 1.5 * p.L;              
            }
            ++iter;
        }
        // std::cout << pred_err.norm() << std::endl;
        return res;
        
    }
    
    
    ///////////////////////////////////////
    // General solvers
    ///////////////////////////////////////

    VectorT ridge(const MatrixT &X, const VectorT &y, g_float lambd2) {
        MatrixT XTy = X.transpose() * y; 
        MatrixT XTX = X.transpose() * X; 
        // add penalty
        XTX.diagonal().array() += lambd2;
        // and solve
        return XTX.colPivHouseholderQr().solve(XTy);
    }
    
    /* first define the fista algorithm */
    // since we are interested in structured sparsity solutions
    // we first define a version that also recovers the group structure
    VectorT group_fista(MatrixT &X, const VectorT &y, const struct params &p, const int N, group_stack &stack) {
        VectorT beta(X.cols());
        VectorT beta_last(X.cols());
        VectorT beta_hat(X.cols());
        g_float tau = 1.;
        g_float tau_last = tau;
        
        VectorT sorted_ind;
        // normalize the columns of X NOTE: this should really be done before calling fista
        //for (int c = 0; c < X.cols(); ++c) {
        //        X.col(c) /= X.col(c).norm();
        //}
        
        //g_float ymean = y.mean();
        //VectorT tmp_y = y;
        //tmp_y.array() -= ymean;
        
        // copy parameters as we might want to adapt L and lambda
        struct params p_cp = p;
        // approximate initial lambda
        //p_cp.lambd1 = 0.5 * (X.transpose() * y).array().abs().maxCoeff();
        // We could use the least squares solution as an initial guess here
        //beta = X.colPivHouseholderQr().solve(y);
        //beta.setRandom();

        if (p.w0.size() > 0) 
            beta = p.w0;
        else
            beta.setZero();
        beta_last = beta;
        beta_hat = beta;
        // add the l2 penalty if possible
        if (X.rows() == X.cols() && p_cp.lambd2 > 0.)
            X.diagonal().array() += p_cp.lambd2;
        int t = 0;
        for (t = 0; t < N; ++t) {
            if (DEBUG >= 0) {
                if (t % 1000 == 0)
                    std::cout << "FISTA iteration " << t << " of " << N << std::endl;
            }
            tau_last = tau;
            tau = (1. + std::sqrt(1. + 4. * SQR(tau_last))) / 2.;
            beta = projection_step(X, y, beta_hat, p_cp, stack, sorted_ind);
            VectorT diff = (beta - beta_last);
            beta_hat = beta + ((tau_last - 1.) / (tau)) * diff;
            // save parameters
            beta_last = beta;
            // update lambda
            //p_cp.lambd1 = std::max(0.95*p_cp.lambd1, lambda_min);
            // stopping condition
            g_float gap = std::sqrt(diff.squaredNorm()) / std::sqrt(std::max(1e-10, beta.squaredNorm()));
            
            if (t % 1000 == 0) {
                std::cout << "Gap after " << t << ": " << gap << std::endl;
            }
            
            if (t % p.it_tol == 0 && gap < p.tol)
                break;
        }
        std::cout << "fista took " << t  << " iterations of " << N << std::endl;
        // DEBUG ONLY
        int groups = 0;
        if (stack.size() > 0) {
            int total_gsize = 0;
            for (group_stack::const_iterator it = stack.begin(); it != stack.end(); ++it) {
                const group &group = it->first;
                if (group.second - group.first > 0) {
                    groups++;
                    total_gsize += (group.second - group.first + 1);
                    std::cout << groups << " size: " << (group.second - group.first + 1) << " value: " << it->second << " real value in beta: " << beta(sorted_ind(group.first)) <<  std::endl;
                }
            }
            std::cout << "Found " << groups << " groups of size > 1 total size: " << total_gsize  << std::endl;
        }
        if (p_cp.refit && p.group_sparse) {                
            std::cout << "starting to refit on reduced problem with l2 regularization: " << p_cp.lambd2d << std::endl;
            ///*
            // refit a standard regression model on the reduced problem 
            // to avoid over penalizing parameters
                int num_col = 0;
                // detect groups to combine
                for (group_stack::const_iterator it = stack.begin(); it != stack.end(); ++it) {
                    const group &group = it->first;
                    int curr_col = sorted_ind(group.first);
                    if (std::fabs(beta(curr_col)) > (p.lambd1 + p.lambdm)) {
                        num_col++;
                        //num_col += group.second - group.first + 1;
                    } 
                }
                std::cout << "total number of columns in reduced problem: " << num_col << std::endl;
                if (num_col == 0) {
                    std::cerr << "WARNING: cannot refit with " << num_col << " columns" << std::endl;
                } else {
                    // remove the l2 penalty
                    if (X.rows() == X.cols() && p_cp.lambd2 > 0.)
                        X.diagonal().array() -= p_cp.lambd2;
                    // build the reduced problem and an inverse mapping
                    MatrixT Xred(X.rows(), num_col);
                    std::map<int, int> mapping;
                    int col_idx = 0;
                    Xred.setZero();
                    // add the l2 penalty
                    //Xred.row(Xred.rows()-1).array() += p_cp.lambd2 * num_col / (X.cols());
                    // combine groups and build other columns
                    for (group_stack::const_iterator it = stack.begin(); it != stack.end(); ++it) {
                        const group &group = it->first;
                        int curr_col = sorted_ind(group.first);
                        if (std::fabs(beta(curr_col)) > (p.lambd1 + p.lambdm)) {
                            if (group.second - group.first > 0) {
                                // combine these columns
                                for (int i = group.first; i <= group.second; ++i) {
                                    curr_col = sorted_ind(i);
                                    mapping[curr_col] = col_idx;
                                    if (beta(curr_col) > 0)
                                        Xred.col(col_idx) += X.col(curr_col);
                                    else
                                        Xred.col(col_idx) -= X.col(curr_col);
                                }
                            } else {
                                // copy this column
                                mapping[curr_col] = col_idx;
                                Xred.col(col_idx) = X.col(curr_col);
                                    }
                            col_idx++;
                        }
                    }
                    // solve the reduced penalized least squares problem
                    VectorT beta_red = ridge(Xred, y, p_cp.lambd2d);
                    // and replace entries in beta
                    for (int i = 0; i < beta.size(); ++i) {
                        if (std::fabs(beta(i)) > (p.lambd1 + p.lambdm)) {
                            beta(i) = beta_red(mapping[i]);
                            }
                    }
                }
        }
        // END DEBUG ONLY
        return beta;
    }
    
    // we then define the normal fista as a special case
    VectorT fista(MatrixT &X, const VectorT &y, const struct params &p, const int N) {
        group_stack g;
        return group_fista(X, y, p, N, g);
    }
}
