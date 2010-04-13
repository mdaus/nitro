/* =========================================================================
 * This file is part of math.poly-c++ 
 * =========================================================================
 * 
 * (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
 *
 * math.poly-c++ is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public 
 * License along with this program; If not, 
 * see <http://www.gnu.org/licenses/>.
 *
 */
#ifndef __MATH_POLY_TWOD_H__
#define __MATH_POLY_TWOD_H__

#include "math/poly/OneD.h"
#include "math/linear/Matrix2D.h"

namespace math
{
namespace poly
{
/*! This class defines an implementation for a 2-D polynomial. It should
    support datatypes that allow the following operators:
    _T(0), _T*_T, _T*=_T, _T+_T, _T+=_T, _T-_T, _T-=_T, -T*<double>

    It supports polynomials of the form:
    a00 + a01*x + a02*x^2 + ...
    y*(a10 + a11*x + a12*x^2 + ...)

    It supports evaluating the integral over a specified interval (a...b, c...d)

    It supports computing the derivative

    And, it supports the multiplication/addtion/subtraction of 2-D polynomials.

    Also note: 
    In a 2-D sense,
       X -> line
       Y -> elem
*/
template<typename _T>
class TwoD
{
protected:
    //! using a vector of one-d polynomials simplify the implementation.
    std::vector<OneD<_T> > mCoef;
    
public:

    std::vector<OneD<_T> >& coeffs(){ return mCoef; }

    TwoD() {}
    TwoD(int orderX, int orderY) : mCoef(orderX+1,OneD<_T>(orderY)) {}
    
    
    template<typename Vector_T> TwoD(int orderX, int orderY, const Vector_T& coeffs)
    {
        mCoef.resize(orderX+1,OneD<_T>(orderY));
        for (int i = 0; i <= orderX; ++i)
        {
            for (int j = 0; j <= orderY; ++j)
            {
                mCoef[i][j] = coeffs[i * (orderY+1) + j];
            }
        }
    }
    int orderX() const { return mCoef.size()-1; }
    int orderY() const { return (orderX() < 0 ? -1 : mCoef[0].order()); }
    _T operator () (double atX, double atY) const;
    _T integrate(double xStart, double xEnd, double yStart, double yEnd) const;

    /*!
     *  Transposes the coefficients so that X is Y and Y is X
     *
     */
    TwoD<_T> flipXY() const;
    TwoD<_T> derivativeY() const;
    TwoD<_T> derivativeX() const;
    TwoD<_T> derivativeXY() const;
    OneD<_T> atY(double y) const;
    OneD<_T> operator [] (unsigned int idx) const;
    /*! In case you are curious about the return value, this guarantees that
      someone can only change the coefficient stored at [x][y], and not the
      polynomial itself. Unfortunately, however, it does not allow one bounds
      checking on the size of the polynomial.
    */
    _T* operator [] (unsigned int idx);
    TwoD<_T>& operator *= (double cv) ;
    TwoD<_T> operator * (double cv) const;
    template<typename _TT>
        friend TwoD<_TT> operator * (double cv, const TwoD<_TT>& p);
    TwoD<_T>& operator *= (const TwoD<_T>& p);
    TwoD<_T> operator * (const TwoD<_T>& p) const;
    TwoD<_T>& operator += (const TwoD<_T>& p);
    TwoD<_T> operator + (const TwoD<_T>& p) const;
    TwoD<_T>& operator -= (const TwoD<_T>& p);
    TwoD<_T> operator - (const TwoD<_T>& p) const;
    TwoD<_T>& operator /= (double cv);
    TwoD<_T> operator / (double cv) const;
    bool operator == (const TwoD<_T>& p) const;
    bool operator != (const TwoD<_T>& p) const;

    TwoD<_T> power(int toThe) const;

    template<typename _TT>
        friend std::ostream& operator << (std::ostream& out, const TwoD<_TT> p);
};

/*!
 *  This is based on Jim W.'s MathCAD for a 2D least squares
 *  fit.  This type of algorithm may be used, for example, to
 *  fit a 2D polynomial plane projection, by picking a grid of
 *  sample coordinates in the planes, ideally sampled with 
 *  roughly twice the number of samples as would be required
 *  to do least squares in a well-tempered case.
 *
 *  To make sure that one dimension does not dominate the other,
 *  we normalize the x and y matrices.
 *
 *  The x, y and z matrices must all be the same size, and the
 *  x(i, j) point in X must correspond to y(i, j) in Y
 *
 *  \param x Input x coordinate
 *  \param y Input y coordinates
 *  \param z Observed outputs
 *  \param nx The requested order X of the output poly
 *  \param ny The requested order Y of the output poly
 *  \throw Exception if matrices are not equally sized
 *  \return A polynomial, f(x, y) = z
 */

inline math::poly::TwoD<double> fit(const math::linear::Matrix2D<double>& x,
				    const math::linear::Matrix2D<double>& y,
				    const math::linear::Matrix2D<double>& z,
				    int nx,
				    int ny)
{
    // Normalize the values in the matrix
    int m = x.rows();
    int n = x.cols();
    
    if (m != y.rows())
        throw except::Exception(Ctxt("Matrices must be equally sized"));

    if (n != y.cols())
        throw except::Exception(Ctxt("Matrices must be equally sized"));
    
    double xacc = 0.0;
    double yacc = 0.0;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            xacc += x(i, j) * x(i, j);
            yacc += y(i, j) * y(i, j);
        }
    }
    
    // by num elements
    int mxn = m*n;
    
    xacc /= (double)mxn;
    yacc /= (double)mxn;
    
    double rxrms = 1/std::sqrt(xacc);
    double ryrms = 1/std::sqrt(yacc);
    
    // Scalar division
    
    math::linear::Matrix2D<double> xp = x * rxrms;
    math::linear::Matrix2D<double> yp = y * ryrms;

    int acols = (nx+1) * (ny+1);

    // R = M x N
    // C = NX+1 x NY+1

    // size(A) = R x P
    math::linear::Matrix2D<double> A(mxn, acols);
    
    for (int i = 0; i < m; i++)
    {
        int xidx = i*n;
        for (int j = 0; j < n; j++)
        {

            // We are doing an accumulation of pow()s to get this

            // Pre-calculate these
            double xij = xp(i, j);
            double yij = yp(i, j);

            xacc = 1;

            for (int k = 0; k <= nx; k++)
            {
                int yidx = k * (ny + 1);
                yacc = 1;
                
                for (int l = 0; l <= ny; l++)
                {

                    A(xidx, yidx) = xacc * yacc;
                    yacc *= yij;
                    ++yidx;

                }
                xacc *= xij;
            }
            // xidx: i*n + j;
            xidx++;
        }
    }
    
    // size(tmp) = R x 1
    math::linear::Matrix2D<double> tmp(mxn, 1);

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            tmp(i*n + j, 0) = z(i, j);
        }
    }
    
    math::linear::Matrix2D<double> At = A.transpose();

    //       T
    // size(A  A) = (P x R) (R x P) = (P x P)
    // size(inv) = (P x P)
    math::linear::Matrix2D<double> inv = math::linear::inverse<double>(At * A);

    // size(C) = ((P x P) (P x R))(R x 1)
    //         =   (P x R)(R x 1)
    //         =   (P x 1)
    //         =   (NX+1xNY+1 x 1)

    math::linear::Matrix2D<double> C = inv * At * tmp;

    // Now we need the NX+1 components out for our x coeffs
    // and NY+1 components out for our y coeffs
    math::poly::TwoD<double> coeffs(nx, ny);

    xacc = 1;
    int p = 0;
    for (int i = 0; i <= nx; i++)
    {
        yacc = 1;
        for (int j = 0; j <= ny; j++)
        {
            coeffs[i][j] = C(p, 0)*(xacc * yacc);
            ++p;
            yacc *= ryrms;
        }
        xacc *= rxrms;
    }
    return coeffs;


}

inline math::poly::TwoD<double> fit(int numRows,
				    int numCols,
				    const double* x,
				    const double* y,
				    const double* z,
				    int nx,
				    int ny)
{
    math::linear::Matrix2D<double> xm(numRows, numCols, x);
    math::linear::Matrix2D<double> ym(numRows, numCols, y);
    math::linear::Matrix2D<double> zm(numRows, numCols, z);

    return fit(xm, ym, zm, nx, ny);
}
 
 

} // poly
} // math
#include "math/poly/TwoD.hpp"
#endif
