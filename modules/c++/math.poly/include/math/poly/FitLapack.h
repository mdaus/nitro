#ifndef __MATH_POLY_FIT_LAPACK_H__
#define __MATH_POLY_FIT_LAPACK_H__


#include <math/poly/OneD.h>
#include <math/poly/TwoD.h>
#include <math/linear/Matrix2D.h>

#include <memory>
#include <cmath>

extern "C" void dgelsd_(
    int*    M,       // in        
    int*    N,       // in
    int*    NRHS,    // in
    const double* A, // in
    int*    LDA,     // in
    double* B,       // in + out
    int*    LDB,     // in
    double* S,       // out
    double* RCOND,   // in
    int*    RANK,    // out
    double* WORK,    // in + out
    int*    LWORK,   // in
    int*    IWORK,   // in + out
    int*    INFO     // out
    );

extern "C" void dgels_(
    const char*   TRANS, // in
    int*    M,     // in
    int*    N,     // in
    int*    NRHS,  // in
    const double* A,     // in + out
    int*    LDA,   // in
    double* B,     // in + out
    int*    LDB,   // in
    double* WORK,  // in + out
    int*    LWORK, // in
    int*    INFO
    );

const int SMLSIZ = 25;

namespace math 
{
namespace poly
{

template<typename Vector_T> OneD<double> fitDgelsd(const Vector_T& x,
                                             const Vector_T& y,
                                             size_t numCoeffs)
{

    size_t sizeX = x.size();

    double* B = new double[sizeX];
    math::linear::Matrix2D<double> A(numCoeffs+1, x.size());
    for( size_t i = 0; i < sizeX; ++i )
    {
        A(0,i) = 1;        
        double v = x[i];
        A(1,i) = v;
        for ( size_t j = 1; j <= numCoeffs; ++j) 
        {
          A(j,i) = std::pow(v, static_cast<double>(j));
        }
        B[i] = y[i];
    }

    int M = A.cols();
    int N = A.rows();
    int NRHS = 1; 
    // already have A
    const double* A_ptr = A.get();
    int LDA = M;
    // already have B
    int LDB = std::max(1,std::max(M,N));
    double* S = new double[std::min(M,N)];
    double RCOND = 0.0;
    int RANK = 0;
    double dtmp = 0.0;
    double* WORK = &dtmp;
    int LWORK = -1; // this signals a query for the workspace size
    int itmp = 0;
    int* IWORK = &itmp;
    int INFO;

    double workspaceSize = 0.0;

    // query to get workspace size
    dgelsd_( &M, &N, &NRHS, A_ptr, &LDA, B, &LDB, S, &RCOND, &RANK,
        &workspaceSize, &LWORK, IWORK, &INFO);

    if(INFO != 0) {
      // error
    }

    LWORK = workspaceSize;
    WORK = new double[LWORK];
    IWORK = new int[LWORK](); // bigger than it needs to be, but what are a few bytes these days
    
    dgelsd_(&M, &N, &NRHS, A_ptr, &LDA, B, &LDB, S, &RCOND, &RANK,
        WORK, &LWORK, IWORK, &INFO);

    if(INFO != 0) {
      // error
    }

    std::vector<double> results(sizeX);
    std::copy(B,B+sizeX,results.begin());

    math::poly::OneD<double> retv(results);

    delete[] S;
    delete[] B;
    delete[] WORK;
    delete[] IWORK;
    return retv;
}

template<typename Vector_T> OneD<double> fitDgels(const Vector_T& x,
                                             const Vector_T& y,
                                             size_t numCoeffs)
{

    size_t sizeX = x.size();

    double* B = new double[sizeX];
    math::linear::Matrix2D<double> A(numCoeffs+1, x.size());
    for( size_t i = 0; i < sizeX; ++i )
    {
        A(0,i) = 1;        
        double v = x[i];
        A(1,i) = v;
        for ( size_t j = 1; j <= numCoeffs; ++j) 
        {
          A(j,i) = std::pow(v, static_cast<double>(j));
        }
        B[i] = y[i];
    }

    int M = A.cols();
    int N = A.rows();
    int NRHS = 1; 
    // already have A
    const double* A_ptr = A.get();
    int LDA = M;
    // already have B
    int LDB = std::max(1,std::max(M,N));
    double dtmp = 0.0;
    double* WORK = &dtmp;
    int LWORK = -1; // this signals a query for the workspace size
    int INFO;

    double workspaceSize = 0.0;

    std::string TRANS = "N";

    // query to get workspace size
    dgels_(TRANS.c_str(), &M, &N, &NRHS, A_ptr, &LDA, B, &LDB, 
        &workspaceSize, &LWORK,&INFO);

    if(INFO != 0) {
      // error
    }

    LWORK = workspaceSize;
    WORK = new double[LWORK];
    
    dgels_(TRANS.c_str(), &M, &N, &NRHS, A_ptr, &LDA, B, &LDB,
        WORK, &LWORK, &INFO);

    if(INFO != 0) {
      // error
    }

    std::vector<double> results(sizeX);
    std::copy(B,B+sizeX,results.begin());

    math::poly::OneD<double> retv(results);

    delete[] B;
    delete[] WORK;
    return retv;
}



/*
 * This function calculates a TwoD polynomial estimating some function
 * f(x,y) = z. It is a more complicated linear least squares fitting 
 * algorithm.
 *
 * Given i observations Z (z_i at point (x_i,y_i)) we want to solve for the 
 * estimation of the product of the polynomial X and Y st X*Y yields Z. This
 * can be done by using each term of the product as independent linear
 * variables, allowing us to use multiple linear regression to solve for the
 * coefficients C (C_0 through C_i^2, since X and Y must be two i-1 order
 * polynomials (remember an n'th order polynomial has n+1 coefficients)).
 *
 * Your typical linear least squares problem looks like Ax=b. (see OneD case)
 * In the "two dimensional" case our A, x, and b are as follows (given the 
 * above input variables)
 *
 * | y_0^0*x_0^0   y_0^0*x_0^1   ...   y_0^1*x_0^0   ...   y_0^i*x_0^i |
 * | y_1^0*x_1^0   y_1^0*x_1^1   ...   y_1^1*x_1^0   ...   y_1^i*x_1^i |
 * |     ...           ...       ...       ...       ...       ...     | = A
 * | y_i^0*x_i^0   y_i^0*x_i^1   ...   y_i^1*x_i^0   ...   y_i^i*x_i^i | 
 *
 * | C_y_0*C_x_0 |
 * | C_y_0*C_x_1 |
 * |     ...     | = x
 * | C_y_1*C_x_0 |
 * |     ...     |
 * | C_y_i*C_x_i |
 *
 * | Z_(y_0,x_0) |
 * | Z_(y_1,x_1) |
 * | Z_(y_2,x_2) | = b
 * |     ...     |
 * | Z_(y_i,x_i) |
 *
 * Thus A is a i row by i^2 column matrix, x is a i^2 row by 1 column matrix,
 * and b is a i row by 1 column matrix.
 *
 */
math::poly::TwoD<double> fit(const math::linear::Matrix2D<double>& x,
    const math::linear::Matrix2D<double>& y,
    const math::linear::Matrix2D<double>& z,
    size_t nx,
    size_t ny)
{
    size_t m = x.rows();
    size_t n = x.cols();
    size_t num_coeffs = m*n;

    math::linear::Matrix2D<double> A(num_coeffs, num_coeffs);

    double pow_x = 0.0;
    double pow_y = 0.0;

    for(size_t i = 0; i < num_coeffs; ++i) 
    {
        for ( size_t j = 1; j <= num_coeffs; ++j) 
        {
            syntax error dont forget to put in pow_x and pow_y 
            A(j,i) = std::pow(x(i,j),pow_x) * std::pow(y(i,j),pow_y);
        }
    }
}

}
}

#endif

