#include <iostream>
#include <iomanip>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

// === Forward/Backward Substitution === //
Vector2d forwardSubstitution(const Matrix2d& L, const Vector2d& b) {
    Vector2d y;
    y(0) = b(0) / L(0, 0);
    y(1) = (b(1) - L(1, 0) * y(0)) / L(1, 1);
    return y;
}

Vector2d backwardSubstitution(const Matrix2d& U, const Vector2d& y) {
    Vector2d x;
    x(1) = y(1) / U(1, 1);
    x(0) = (y(0) - U(0, 1) * x(1)) / U(0, 0);
    return x;
}

// === PALU Solver === //
Vector2d solveWithPALU(const Matrix2d& A, const Vector2d& b) {
    PartialPivLU<Matrix2d> lu(A);
    Matrix2d L = Matrix2d::Identity();
    Matrix2d U = lu.matrixLU().triangularView<Upper>();
    L.triangularView<StrictlyLower>() = lu.matrixLU().triangularView<StrictlyLower>();
    PermutationMatrix<2> P = lu.permutationP();

    Vector2d permuted_b = P * b;
    Vector2d y = forwardSubstitution(L, permuted_b);
    return backwardSubstitution(U, y);
}

// === QR Solver === //
Vector2d solveWithQR(const Matrix2d& A, const Vector2d& b) {
    HouseholderQR<Matrix2d> qr(A);
    Matrix2d Q = qr.householderQ();
    Matrix2d R = qr.matrixQR().triangularView<Upper>();

    Vector2d y = Q.transpose() * b;
    return backwardSubstitution(R, y);
}

// === Utility: Relative Error === //
double relative_error(const Vector2d& my_ans, const Vector2d& ans) {
    return (my_ans - my_ans).norm() / ans.norm();
}

// === system solver === //
void analyzeLinearSystem(const Matrix2d& A, const Vector2d& b, const Vector2d& expected_solution, int system_id) {
    cout << "=== System " << system_id << " ===\n";
    cout << "Matrix A:\n" << A.format(IOFormat(10, 0, ", ", "\n", "", "", "", "")) << "\n";
    cout << "Vector b: " << b.transpose().format(IOFormat(10, 0, ", ", "", "", "", "", "")) << "\n\n";

    Vector2d solution_palu = solveWithPALU(A, b);
    double error_palu = relative_error(solution_palu, expected_solution);

    Vector2d solution_qr = solveWithQR(A, b);
    double error_qr = relative_error(solution_qr, expected_solution);

    cout << "PALU Solution:       " << solution_palu.transpose().format(IOFormat(10, 0, ", ", "", "", "", "", "")) << "\n";
    cout << "PALU Relative Error: " << error_palu << "\n";

    cout << "QR Solution:         " << solution_qr.transpose().format(IOFormat(10, 0, ", ", "", "", "", "", "")) << "\n";
    cout << "QR Relative Error:   " << error_qr << "\n";

    cout << "-----------------------------\n\n";
}

// === Main Function === //
int main() {
    // Set global scientific notation and precision
    cout << scientific << setprecision(10);

    Vector2d expected_solution(-1.0, -1.0);

    Matrix2d A1, A2, A3;
    Vector2d b1, b2, b3;

    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
          8.320502943378437e-01, -9.992887623566787e-01;
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;

    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
          8.320502943378437e-01, -8.324762492991313e-01;
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;

    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
          8.320502943378437e-01, -8.320502947645361e-01;
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;

    analyzeLinearSystem(A1, b1, expected_solution, 1);
    analyzeLinearSystem(A2, b2, expected_solution, 2);
    analyzeLinearSystem(A3, b3, expected_solution, 3);

    return 0;
}
