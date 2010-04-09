/* =========================================================================
 * This file is part of math.linear-c++ 
 * =========================================================================
 * 
 * (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
 *
 * math.linear-c++ is free software; you can redistribute it and/or modify
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
#ifndef __MATH_LINEAR_MATRIX_2D_H__
#define __MATH_LINEAR_MATRIX_2D_H__

#include <cmath>
#include <algorithm>
#include <import/sys.h>
#include "math/linear/MatrixMxN.h"

namespace math
{
namespace linear
{

template <typename _T=double>
class Matrix2D
{
public:
    size_t mM;
    size_t mN;
    std::vector<_T> mRaw;

    //! Default constructor (does nothing)
    Matrix2D() { mM = mN = 0; }

    /*!
     *  Set constant value embedded across matrix
     *
     *  \code
           Matrix2D<double> mx(4.2, 2, 2);
     *  \endcode
     */
    Matrix2D(size_t M, size_t N)
    {
        mRaw.resize(M * N, 0);
        mM = M;
        mN = N;
    }

    /*!
     *  Create a matrix with a constant value for
     *  each element.
     *
     *  \param cv A constant value (defaults to 0)
     *
     *  \code
           Matrix2D<float> mx(2, 2, 4.2f);
     *  \endcode
     *
     */
    Matrix2D(size_t M, size_t N, _T cv)
    {

        mRaw.resize(M * N, cv);
        mM = M;
        mN = N;
    }


    //! Nothing is allocated
    ~Matrix2D() {}


    /*!
     *  Operator with compile time size check only
     *  \param i The ith index into the rows (M)
     *  \param j The jth index into the cols (N)
     */
    inline _T operator()(int i, int j) const
    {
#if defined(MATH_LINEAR_BOUNDS)
        assert( i < mM && j < mN );
#endif
        return mRaw[i * mN + j];
    }

    /*!
     *  This operator allows you to mutate an element
     *  at mx(i, j):
     *  
     *  \code
           mx(i, j) = 4.3;
     *  \endcode
     *
     *  \param i The ith index into the rows (M)
     *  \param j The jth index into the cols (N)
     */
    inline _T& operator()(int i, int j)
    {
#if defined(MATH_LINEAR_BOUNDS)
        assert( i < mM && j < mN );
#endif
        return mRaw[i * mN + j];
    }

    /*!
     *  Supports explicit assignment from
     *  one matrix to another
     *
     *  \param mx
     *
     */
    Matrix2D(const Matrix2D& mx)
    {
        mRaw = mx.mRaw;
        mM = mx.mN;
        mN = mx.mN;
    }

    Matrix2D(size_t M, size_t N, const std::vector<_T>& raw)
    {
        mRaw = raw;
        mM = M;
        mN = N;
    }

    Matrix2D(size_t M, size_t N, const _T* raw)
    {
        size_t sz = M * N;
        mRaw.resize(sz);
        for (unsigned int i = 0; i < sz; ++i)
        {
           mRaw[i] = raw[i];
        }
        mM = M;
        mN = N;
    }
    /*!
     *  Assignment operator from one matrix to another
     *
     *  \param mx The source matrix
     *  \return this (the copy)
     */
    Matrix2D& operator=(const Matrix2D& mx)
    {
        if (this != &mx)
        {
            mRaw = mx.mRaw;
            mM = mx.mM;
            mN = mx.mN;
        }
        return *this;
    }

        
    template<typename Matrix_T> inline bool operator==(const Matrix_T& mx) const
    {
        
        if (rows() != mx.rows() || cols() != mx.cols())
            return false;

        size_t M = rows();
        size_t N = cols();
        for (size_t i = 0; i < M; ++i)
        {
            for (size_t j = 0; j < N; ++j)
            {
                if (! equals(mRaw[i * N + j], mx(i, j)))
                    return false;
            }
        }
        return true;
    }
          
    template<typename Matrix_T> inline bool operator!=(const Matrix_T& mx) const
    {
            return !(*this == mx);
    }
    /*
    bool operator==(const Matrix2D& mx) const
    {
        if (this != &mx)
        {
            if (mx.mM != mM || mx.mN != mN)
                return false;

            for (size_t i = 0; i < mRaw.size(); ++i)
            {
                if (!equals(mRaw[i], mx.mRaw[i]))
                    return false;
            }
        }
        return true;
    }
    bool operator!=(const Matrix2D& mx) const
    {
        return !(*this == mx);
    }*/
    /*!
     *  Set a matrix to a single element containing the contents
     *  of a scalar value:
     *
     *  \code
           mx = 4.3f;
     *  \endcode
     *
     */
    Matrix2D& operator=(const _T& sv)
    {
        mM = 1;
        mN = 1;
        mRaw.resize(1, sv);
        return *this;
    }

    //!  Return number of rows (M)
    size_t rows() const { return mM; }

    //!  Return number of cols (N)
    size_t cols() const { return mN; }

    //!  Return total size (M x N)
    size_t size() const { return mRaw.size(); }

    /*!
     *  The scale function allows you to scale
     *  the matrix by a scalar value in-place.
     *  
     *  If you can afford to mutate the matrix,
     *  this will be more efficient than its
     *  multiply counterpart
     *
     *  \code
           mx.scale(4.2f);   
     *  \endcode
     *
     *
     *  \param scalar The value to multiply into mx
     *
     */
    void scale(_T scalar)
    {
        size_t sz = mRaw.size();
        for (size_t i = 0; i < sz; ++i)
            mRaw[i] *= scalar;
    }

    /*!
     *  Does a matrix-scalar multiplication.  The
     *  output is a matrix of same dimensions.  This
     *  function is identical to the scale() method
     *  except that this matrix is not mutated -
     *  a scale copy is produced and returned
     *
     *  \param
     *  
     *  \code
           scaled = mx.multiply(scalar);
     *  \endcode
     *
     *  return a scaled matrix
     *
     */
    Matrix2D multiply(_T scalar) const
    {
        size_t sz = mRaw.size();
        Matrix2D mx = *this;
       
        for (size_t i = 0; i < sz; ++i)
        {
                mx.mRaw[i] *= scalar;
        }
        return mx;
    }

    /*!
     *  Multiply an NxP matrix to a MxN matrix (this) to
     *  produce an MxP matrix output.
     *
     *  This function accesses the inner arrays for
     *  (potential, though slight) performance reasons.
     *
     *  One would hope that the compiler will unroll these
     *  loops since they are of constant size.
     *
     *  \param mx An NxP matrix
     *  \return An MxP matrix
     *
     *
     *  \code
           C = B.multiply(A);
     *  \endcode
     *
     */
    Matrix2D
        multiply(const Matrix2D& mx) const
    {
        if (mN != mx.mM)
            throw except::Exception(Ctxt("Invalid size for multiply"));

        unsigned int M = mM;
        unsigned int N = mN;
        unsigned int P = mx.mN;

        Matrix2D newM(M, P);
        
        unsigned int i, j, k;

        for (i = 0; i < M; i++)
        {
            for (j = 0; j < P; j++)
            {
		        newM(i, j) = 0;

                for (k = 0; k < N; k++)
                {
                    newM(i, j) += mRaw[i * N + k] * mx(k, j);
                }
            }
        }
        return newM;

    }


    /*!
     *  Take in a matrix that is NxN and apply each diagonal
     *  element to the column vector.  This method mutates this,
     *  but returns it out as well.
     *
     *  For each column vector in this, multiply it by the scalar
     *  diagonal value then assign the result.
     *
     *  \param mx An NxN matrix whose diagonals scale the columns
     *  \return A reference to this.
     *
     *  \code
           C = A.scaleDiagonal(diagonalMatrix);
     *  \endcode
     *
     *
     */
    Matrix2D& scaleDiagonal(const Matrix2D& mx)
    {
        if (mx.mM != mx.mN || mx.mN != mN)
            throw except::Exception(Ctxt("Invalid size for diagonal multiply"));

        unsigned int i, j;
        for (i = 0; i < mM; i++)
        {
            for (j = 0; j < mN; j++)
            {
                mRaw[i * mN + j] *= mx(j,j);
            }
        }
        return *this;
    }

    /*!
     *  This function is the same as scaleDiagonal except that
     *  it does not mutate this (it makes a copy and then calls
     *  that function on the copy).
     *
     *  \param mx An NxN matrix whose diagonals scale the columns
     *  \return a copy matrix
     *  
     *  \code
           C = A.multiplyDiagonal(diagonalMatrix);
     *  \endcode
     *
     *
     */
    Matrix2D
        multiplyDiagonal(const Matrix2D& mx) const
    {
        Matrix2D newM = *this;
        newM.scaleDiagonal(mx);
        return newM;
    }

    /*!
     *  This function does an add and accumulate
     *  operation.  The parameter is add-assigned
     *  element-wise to this
     *
     *  \param mx The matrix to assign (MxN)
     *  \return This
     *     
     *  \code
           A += B;
     *  \endcode
     *
     */
    Matrix2D&
    operator+=(const Matrix2D& mx)
    {
        if (mM != mx.mM || mN != mx.mN)
            throw except::Exception(Ctxt("Required to equally size matrices for element-wise add"));

        size_t sz = mRaw.size();
        for (size_t i = 0; i < sz; ++i)
        {
            mRaw[i] += mx.mRaw[i];
        }
        return *this;

    }

    /*!
     *  This function does a subtraction
     *  operation element wise.
     *
     *  \param mx MxN matrix to subtract from this
     *  \return This
     *
     *  \code
           A -= B;
     *  \endcode
     *
     */
    Matrix2D&
    operator-=(const Matrix2D& mx)
    {
        if (mx.mM != mM || mx.mN != mN)
            throw except::Exception(Ctxt("Matrices must be same size for element-wise subtract"));
        
        for (unsigned int i = 0; i < mM; i++)
        {
            for (unsigned int j = 0; j < mN; j++)
            {
                mRaw[i * mN + j] -= mx(i, j);
            }
        }
        return *this;

    }

    /*!
     *  Add an MxN matrix to another and return a third
     *  that is the sum.  This operation does not mutate this.
     *
     *  \param mx
     *  \return The sum
     *
     *  \code
           C = A.add(B);
     *  \endcode
     *
     */
    Matrix2D add(const Matrix2D& mx) const
    {
        Matrix2D newM = *this;
        newM += mx;
        return newM;
    }

    /*!
     *  Subtract an MxN matrix to another and return a third
     *  that is the sum.  This operation does not mutate this.
     *
     *  \param mx
     *  \return The sum
     *
     *  \code
           C = A.subtract(B);
     *  \endcode
     *
     */

    Matrix2D subtract(const Matrix2D& mx) const
    {
        Matrix2D newM = *this;
        newM -= mx;
        return newM;
    }

    /*!
     * Please try never to use this method.  There is extensive information
     * here about why you should avoid it:
     * http://www.parashift.com/c++-faq-lite/operator-overloading.html#faq-13.10
     * http://www.parashift.com/c++-faq-lite/operator-overloading.html#faq-13.11
     */
    inline const _T* operator[](int i) const
    {
        return row(i);
    }

    inline _T* operator[](int i)
    {
        return row(i);
    }


    inline const _T* row(int i) const
    {
#if defined(MATH_LINEAR_BOUNDS)
        assert( i < mM);
#endif
        return &mRaw[i * mN];
    }

    inline _T* row(int i)
    {
#if defined(MATH_LINEAR_BOUNDS)
        assert( i < mM);
#endif
        return &mRaw[i * mN];
    }

    std::vector<_T> col(size_t j) const
    {
        std::vector<_T> jth(mM);
        for (unsigned int i = 0; i < mM; ++i)
        {
            jth[i] = mRaw[i * mN + j];
        }
        return jth;
    }

    /*!
     *  Create a NxM matrix which is the transpose of this
     *  MxN matrix.
     *
     *  \return An NxM matrix that is the transpose
     *
     *  \code
           B = A.tranpose();
     *  \endcode
     *
     */
    Matrix2D transpose() const
    {

        Matrix2D x(mN, mM);
        for (int i = 0; i < mM; i++)
            for (int j = 0; j < mN; j++)
                x.mRaw[j * mM + i] = mRaw[i * mN + j];

        return x;
    }

    /*!
     *  Does LU decomposition on a matrix.
     *  In order to do this efficiently, we get back
     *  as a return value a copy of the matrix that is
     *  decomposed, but we also produce the pivots for
     *  permutation.  This function is used for the generalized
     *  inverse.
     *
     *  This function is based on the TNT LU decomposition
     *  function.
     *
     *  \param [out] pivotsM (pre sized)
     *
     */
    Matrix2D decomposeLU(std::vector<int>& pivotsM) const
    {

        Matrix2D lu(mM, mN);

        for (unsigned int i = 0; i < mM; i++)
        {
            // Start by making our pivots unpermuted
            pivotsM[i] = i;
            for (unsigned int j = 0; j < mN; j++)
            {
                // And copying elements
                lu.mRaw[i * mN + j] = mRaw[i * mN + j];
            }
        }

        std::vector<_T> colj(mM);
        _T* rowi;

        for (unsigned int j = 0; j < mN; j++)
        {
            for (unsigned int i = 0; i < mM; i++)
            {
                colj[i] = lu(i, j);
            }

            for (unsigned int i = 0; i < mM; i++)
            {
                rowi = lu[i];

                int max = std::min<int>(i, j);
                double s(0);
                for (int k = 0; k < max; k++)
                {
                    s += rowi[k] * colj[k];
                }
                colj[i] -= s;
                rowi[j] = colj[i];

            }

            unsigned int p = j;
            for (unsigned int i = j + 1; i < mM; i++)
            {
                if (std::abs(colj[i]) > std::abs(colj[p]))
                    p = i;

            }
            if (p != j)
            {
                unsigned int k = 0;
                for (; k < mN; k++)
                {
                    // We are swapping
                    double t = lu(p, k);
                    lu(p, k) = lu(j, k);
                    lu(j, k) = t;
                }
                k = pivotsM[p];
                pivotsM[p] = pivotsM[j];
                pivotsM[j] = k;

                if (j < mM && lu(j, j) )
                {
                    for (unsigned int i = j + 1; i < mM; i++)
                    {
                        // Divide out our rows
                        lu(i, j) /= lu(j, j);
                    }
                }
            }


        }

        return lu;
    }

     /*!
     *  Cleans up a Matrix where values are within epsilon of zero
     *
     *
     */
    void tidy(_T eps = std::numeric_limits<_T>::epsilon())
    {
        size_t sz = mM * mN;
        for (unsigned int i = 0; i < sz; ++i)
        { 
            if (equals<_T>(mRaw[i], 0, eps))
            {
                mRaw[i] = 0;
                
            }
        }
    }
    /*!
     *  Permute a matrix from pivots.  This funtion does
     *  not mutate this.
     *
     *  \param pivotsM The M pivots vector
     *  \param n The number of columns (defaults to this' N)
     *  \return A copy of this matrix that is permuted
     *
     *  \code
           int p[2] = { 1, 0 };
           Matrix2D G = F.permute(P);
     *  \endcode
     *
     */
    Matrix2D permute(const std::vector<int>& pivotsM, int n = -1) const
    {
        if (n == -1) n = mN;
        Matrix2D perm(mM, n);
        for (unsigned int i = 0; i < mM; i++)
        {
            for (unsigned int j = 0; j < (unsigned int)n; j++)
            {
                perm(i, j) = mRaw[pivotsM[i] * mN + j];
            }
        }
        return perm;
    }

    _T norm() const
    {
        size_t sz = mM * mN;
        _T acc(0);
        for (unsigned int i = 0; i < sz; ++i)
        {
            acc += mRaw[i] * mRaw[i];
        }
        return (_T)::sqrt((const _T)acc);
    }

    void normalize()
    {
        scale(1.0/norm());
    }
    /*!
     *  Alias for this->add();
     *
     *  \code
           C = A + B;
     *  \endcode
     *
     */
    Matrix2D operator+(const Matrix2D& mx) const
    {
        return add(mx);
    }


    /*!
     *  Alias for this->subtract();
     *
     *  \code
           C = A - B;
     *  \endcode
     *
     */
    Matrix2D operator-(const Matrix2D& mx) const
    {
        return subtract(mx);
    }

    /*!
     *  Alias for this->multiply(scalar);
     *
     *  \code
           scaled = A * scalar;
     *  \endcode
     *
     */
    Matrix2D operator*(_T scalar) const
    {

        return multiply(scalar);
    }

    /*!
     *  Alias for this->multiply(1/scalar);
     *
     *  \code
           scaled = A / scalar;
     *  \endcode
     *
     */
    Matrix2D operator/(_T scalar) const
    {

        return multiply(1/scalar);
    }

    /*!
     *  Alias for this->multiply(NxP);
     *
     *  \code
           C = A * B;
     *  \endcode
     */
    Matrix2D
    operator*(const Matrix2D& mx) const
    {
        return multiply(mx);
    }


};

/*!
 *  This function creates an identity matrix of size NxN and type _T
 *  with 1's in the diagonals.
 *
 *  \code
        Matrix2D = identityMatrix<double>(4);
 *  \endcode
 *
 */
template<typename _T> Matrix2D<_T>
    identityMatrix(size_t N)
{
    Matrix2D<_T> mx(N, N);
    for (unsigned int i = 0; i < N; i++)
    {
        for (unsigned int j = 0; j < N; j++)
        {
            mx(i, j) = (i == j) ? 1: 0;
        }
    }
    return mx;
}




/*!
 *  Solve  Ax = b using LU decomposed matrix and the permutation vector.
 *  Method based on TNT
 *
 */
template<typename _T>
    math::linear::Matrix2D<_T> solveLU(const std::vector<int>& pivotsM,
                                       const Matrix2D<_T> &lu,
                                       const Matrix2D<_T> &b)
{


    // If we dont have something in the diagonal, we can't solve this
    math::linear::Matrix2D<_T> x = b.permute(pivotsM);

    unsigned int P = b.mN;
    unsigned int N = lu.mN;
    for (unsigned int k = 0; k < N; k++) {
        for (unsigned int i = k + 1; i < N; i++) {
            for (unsigned int j = 0; j < P; j++) {
                x(i, j) -= x(k, j)*lu(i, k);
            }
        }
    }
    for (int k = N - 1; k >= 0; k--) {
        for (unsigned int j = 0; j < P; j++) {
            x(k, j) /= lu(k, k);
        }

        for (unsigned int i = 0; i < (unsigned int)k; i++) {
            // This one could be _Q
            for (unsigned int j = 0; j < P; j++) {
                x(i, j) -= x(k, j)*lu(i, k);
            }
        }
    }

    return x;
}

    
template<typename _T> inline Matrix2D<_T> inverse2x2(const Matrix2D<_T>& mx)
{
    Matrix2D<_T> inv(2, 2);
    double determinant = mx(1,1) * mx(0,0) - mx(1,0)*mx(0,1);
    
    if (equals(determinant, 0.0))
        throw except::Exception(Ctxt("Non-invertible matrix!"));

    // Standard 2x2 inverse
    inv(0,0) =  mx(1,1);
    inv(0,1) = -mx(0,1);
    inv(1,0) = -mx(1,0);
    inv(1,1) =  mx(0,0);

    inv.scale( 1.0 / determinant );
    return inv;
}

template<typename _T> inline Matrix2D<_T> 
    inverse3x3(const Matrix2D<_T>& mx)
{
    Matrix2D<double> inv(3, 3);

    double a = mx(0,0);
    double b = mx(0,1);
    double c = mx(0,2);

    double d = mx(1,0);
    double e = mx(1,1);
    double f = mx(1,2);

    double g = mx(2,0);
    double h = mx(2,1);
    double i = mx(2,2);

    double g1 = e*i - f*h;
    double g2 = d*i - f*g;
    double g3 = d*h - e*g;

    double determinant = 
        a*g1 - b*g2 + c*g3;
    
    if (equals(determinant, 0.0))
        throw except::Exception(Ctxt("Non-invertible matrix!"));


    inv(0,0) =  g1; inv(0,1) =  c*h - b*i; inv(0,2) =  b*f - c*e;
    inv(1,0) = -g2; inv(1,1) =  a*i - c*g; inv(1,2) =  c*d - a*f;
    inv(2,0) =  g3; inv(2,1) =  b*g - a*h; inv(2,2) =  a*e - b*d;
    inv.scale( 1.0 / determinant );
    
    return inv;

}


/*!
 *  Generalized inverse method (currently uses LU decomposition).
 *
 *  \param mx A matrix to invert
 *
 *  \code
         inv = inverse(A);
 *  \endcode
 *
 */
template<typename _T> inline
    Matrix2D<_T> inverseLU(const Matrix2D<_T>& mx)
{
   

    unsigned int M = mx.mM;
    unsigned int N = mx.mN;
    Matrix2D<_T> a(M, M, (_T)0);

    for (unsigned int i = 0; i < M; i++)
        a(i, i) = 1;

    std::vector<int> pivots(M);
    Matrix2D<_T> lu = mx.decomposeLU(pivots);
    
    for (unsigned int i = 0; i < N; i++)
    {
        if ( equals<_T>(lu(i, i), 0) )
            throw except::Exception(Ctxt("Non-invertible matrix!"));
    }

    return solveLU(pivots, lu, a);

}

template<typename _T> inline
    Matrix2D<_T> inverse(const Matrix2D<_T>& mx)
{
    // Try to speed this up
    if (mx.mM != mx.mN)
        throw except::Exception(Ctxt("Expected a square matrix"));
    if (mx.mM == 2)
        return inverse2x2<_T>(mx);
    if (mx.mM == 3)
        return inverse3x3<_T>(mx);
        // TODO Add 4x4
    
    return inverseLU<_T>(mx);
}
}
}

template<typename _T> math::linear::Matrix2D<_T>
    operator*(_T scalar, const math::linear::Matrix2D<_T>& m)
{
    return m.multiply(scalar);
}


template<typename _T>
    std::ostream& operator<<(std::ostream& os,
                             const math::linear::Matrix2D<_T>& m)
{


    unsigned int i, j;
    std::cout << "(" << m.mM << ',' << m.mN << ")" << std::endl;
    for (i = 0; i < m.mM; ++i)
    {
        for (j = 0; j < m.mN; ++j)
        {
            os << std::setw(10) << m(i, j) << " ";
        }
        os << std::endl;
    }


    return os;
}


#endif
