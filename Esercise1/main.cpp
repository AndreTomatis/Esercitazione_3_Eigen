#include <iostream>
#include "Eigen/Eigen"


using namespace std;
using namespace Eigen;


// shape(n) is the n-by-n matrix with elements from 1 to n * n.
MatrixXd Shape(const unsigned int& n)
{

    VectorXd linspaceVector = VectorXd::LinSpaced(n * n, 1, n * n);
    MatrixXd A = linspaceVector.reshaped<RowMajor>(n, n);

    cout << "Shape(n): \n" << A << endl;

    return A;
}

// rand(n) is the n-by-n matrix with random elements.
MatrixXd Rand(const unsigned int& n)
{

    srand(5);

    // Random: Numbers are uniformly spread and in the [-1,1] range for floating point scalar
    MatrixXd A = MatrixXd::Random(n, n) + MatrixXd::Constant(n, n, 1);

    cout << "Rand(n): \n" << A << endl;
    return A;
}

// hilb(n) is the n-by-n matrix with elements 1/(i+j-1).
MatrixXd Hilb(const unsigned int& n)
{

    MatrixXd A = MatrixXd::Zero(n, n);

    for(unsigned int i = 0; i < n; i++)
        A.col(i) = VectorXd::LinSpaced(n, i + 1, n + i);

    MatrixXd H = A.cwiseInverse();

    cout << "Hilb(n): \n" << H << endl;

    return H;
}

/* MatrixIsSingular(A) compute the determinant and the condition number of A,
*  determining if A is invertible or not
*  detA the determinant of A
*  condA the condition number of A
*  return true if the matrix A is invertible, else it returns false */
bool MatrixIsSingular(const MatrixXd& A,
                      double& detA,
                      double& condA)
{
    JacobiSVD<MatrixXd> svd(A);
    VectorXd singularValuesA = svd.singularValues();
    condA = singularValuesA.maxCoeff() / singularValuesA.minCoeff();

    detA = A.determinant();

    if( singularValuesA.minCoeff() < 1e-16)
        return false;
    else
        return true;
}

int main()
{
    int n = 4;

    double detAS, condAS;
    if(MatrixIsSingular(Shape(n), detAS, condAS))
        cout<< scientific<< "shape - DetA: "<< detAS << ", RCondA: " << 1.0 / condAS << endl;
    else
        cout << scientific<< "shape - DetA: "<< detAS << ", RCondA: " << 1.0 / condAS << " (Matrix is singular)"<< endl;

    double detAR, condAR;
    if(MatrixIsSingular(Rand(n), detAR, condAR))
        cout<< scientific<< "rand - DetA: "<< detAR << ", RCondA: " << 1.0 / condAR << endl;
    else
        cout<< scientific<< "rand - DetA: "<< detAR << ", RCondA: " << 1.0 / condAR << " (Matrix is singular)"<< endl;

    double detAH, condAH;
    if(MatrixIsSingular(Hilb(n), detAH, condAH))
        cout<< scientific<< "hilb - DetA: "<< detAH<< ", RCondA: "<< 1.0 / condAH << endl;
    else
        cout<< scientific<< "hilb - DetA: "<< detAH<< ", RCondA: "<< 1.0 / condAH << " (Matrix is singular)"<< endl;

    return 0;
}

