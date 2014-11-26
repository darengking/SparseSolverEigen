#include <iostream>
#include <sparse_solve.hpp>

using namespace ssolver;

int main(int argc, char *argv[]) {
        std::cout << "Preparing problem" << std::endl;
        srand(123);
        
        Eigen::MatrixXd A;
        Eigen::VectorXd w_real;
        Eigen::VectorXd y_real; 
        
        const int size = 50;
        const int samples = 200;
        const int num_zeros = size / 3;
        A.setRandom(samples, size);
        //A *= 0.01;
        w_real.setRandom(size);
        
        // make the signal sparse
        for (int i = 0; i < num_zeros; ++i) {
                
                int r = rand() / ( RAND_MAX / size + 1 );
                w_real(r) = 0.;
        }
        
        y_real = A * w_real;
        
        // run the fista version of oscar
        std::cout << "starting the optimization" << std::endl;
        struct params p;
        p.lambdm = 1e-5;
        p.lambd1 = 0.;//1e-5;
        p.lambd2 = 0.;
        p.L = 1;
        
        Eigen::VectorXd w = fista(A, y_real, p, 50);
        std::cout << "real w" << std::endl << w_real.transpose() << std::endl;
        //std::cout << "predicted w" << std::endl << w.transpose() << std::endl;
        for (int i = 0; i < w.size(); ++i)
                std::cout << i << ":" << w(i) << " "; 
        std::cout << std::endl;
}
