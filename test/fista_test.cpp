#include <iostream>
#include <sparse_solve.hpp>
#include <fstream>

using namespace ssolver;

int read_matrix_from_file(std::string filename, Eigen::MatrixXd & matrix) 
{
        std::ifstream fin;
        fin.open (filename);
        if (fin.fail())
                {
                         std::cout << "Input file " << filename << ": opening failed." << std::endl;
                         exit(1);
                }

        int m, n;
        fin >> m;
        fin >> n;
                       
        matrix.setZero(m,n);

        for (int i = 0; i < m; i++) {
                for(int j = 0; j < n; j++) {
                        fin >> matrix(i,j);
                }
        }
        return 1;
}

int read_vector_from_file(std::string filename, Eigen::VectorXd & vec) 
{
        std::ifstream fin;
        fin.open (filename);
        if (fin.fail())
                {
                        std::cout << "Input file " << filename << ": opening failed." << std::endl;
                        exit(1);
                }

        // read vector size from first line and set vector to correct size
        int m;
        fin >> m;
        vec.setZero(m);

        for (int i = 0; i < m; i++) {
                fin >> vec(i);
        }
        return 1;
}

int main() {
        int iterations = 200;

        struct params sparse_params;
        sparse_params.lambd1 = 0.05;
        sparse_params.lambdm = 0;
        sparse_params.L = 1;
        sparse_params.tol = 1e-3;
        sparse_params.it_tol = 10;
        sparse_params.lambd2 = 0;
        sparse_params.lambd2d = 0;
        sparse_params.refit = false;

        Eigen::MatrixXd A;
        Eigen::VectorXd y;
        Eigen::VectorXd x_f;
       
        std::cout << "NOTE please start this binary from the test subfolder" << std::endl;
        read_matrix_from_file("data/A.emat", A);
        read_vector_from_file("data/y.evec", y);
        read_vector_from_file("data/xf.evec", x_f);

        
        // do some fista here
        VectorT w = fista(A, y, sparse_params, iterations);

        // TODO: compare solutions
        std::cout << "w size: " << w.size() << ", x_f size: " << x_f.size() << std::endl;
        double dist = (w - x_f).norm();
        std::cout << "Distance: " << dist << std::endl;
        std::cout << "Error: " <<  (y -  A * w).norm() << std::endl;
        
}
