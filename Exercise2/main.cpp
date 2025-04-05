#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

// Function to compute and print the solution and relative error
void solveSystem(const Matrix2d& A, const Vector2d& b, const Vector2d& x_expected) {
    cout << "Matrix A:\n" << A << "\n";
    cout << "Vector b:\n" << b.transpose() << "\n\n";

    // Solve using PartialPivLU (PALU)
    Vector2d x_palu = A.partialPivLu().solve(b);
    double error_palu = (x_palu - x_expected).norm() / x_expected.norm();

    cout << "PALU solution x:\n" << x_palu.transpose() << "\n";
    cout << "PALU relative error: " << error_palu << "\n";

    // Solve using Householder QR
    Vector2d x_qr = A.householderQr().solve(b);
    double error_qr = (x_qr - x_expected).norm() / x_expected.norm();

    cout << "QR solution x:\n" << x_qr.transpose() << "\n";
    cout << "QR relative error: " << error_qr << "\n";

    cout << "----------------------------------------\n";
}

int main() {
    Vector2d x_expected;
    x_expected << -1.0, -1.0;

    // System 1
    Matrix2d A1;
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
          8.320502943378437e-01, -9.992887623566787e-01;
    Vector2d b1;
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;
    solveSystem(A1, b1, x_expected);

    // System 2
    Matrix2d A2;
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
          8.320502943378437e-01, -8.324762492991313e-01;
    Vector2d b2;
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;
    solveSystem(A2, b2, x_expected);

    // System 3
    Matrix2d A3;
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
          8.320502943378437e-01, -8.320502947645361e-01;
    Vector2d b3;
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;
    solveSystem(A3, b3, x_expected);

    return 0;
}
