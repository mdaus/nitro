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

#ifndef __MATH_POLY_ONED_H__
#define __MATH_POLY_ONED_H__

#include <iostream>
#include <sstream>
#include <vector>
#include <math/linear/Vector.h>

namespace math
{
namespace poly
{
/*! 
 *  \class OneD
 *  \brief 1-D polynomial evaluation
 *
 *  This class defines an implementation for a 1-D polynomial. It should
 *  support datatypes that allow the following operators:
 *  _T(0), _T*_T, _T*=_T, _T+_T, _T+=_T, _T-_T, _T-=_T, -T*<double>
 *
 *  It supports polynomials of the form:
 *  a0 + a1*x + a2*x^2 + ... (in ascending powers)
 *
 *   It supports evaluating the integral over a specified interval (a...b)
 *
 *   It supports computing the derivative and 
 *   the multiplication/addition/subtraction of 1-D polynomials.
 */
template<typename _T>
class OneD
{
protected:
    std::vector<_T> mCoef;
    
public:
    OneD() {}
    
    /*!
     *  A vector of ascending power coefficients (note that
     *  this is the reverse of Matlab)
     */
    OneD(const std::vector<_T>& coef) : mCoef(coef) {}

    /*!
     *  Create a vector of given order, with each coefficient
     *  set to zero
     */
    OneD(size_t order)  
    {
	mCoef.resize(order + 1,0); 
    }
    /*!
     *  This function allows you to copy the values
     *  directly from a raw buffer.  The first argument
     *  is the order, NOT THE NUMBER OF COEFFICIENTS.
     *  Therefore, you should always pass an array with
     *  one more element than the first argument
     *
     *  The power coefficients are in ascending order
     *  (note that this is the reverse of Matlab)
     *
     *  \param order The order of the polynomial
     *  \param The order + 1 coefficients to initialize
     */
    OneD(size_t order, const _T* coef)
    {
	mCoef.resize(order + 1);
	memcpy(&mCoef[0], coef, (order + 1) * sizeof(_T));
    }
    int order() const { return mCoef.size()-1; }

    inline size_t size() const { return mCoef.size(); }

    _T operator () (double at) const;
    _T integrate(double start, double end) const;
    OneD<_T> derivative() const;
    _T& operator [] (unsigned int idx);
    _T operator [] (unsigned int idx) const;
    template<typename _TT>
        friend std::ostream& operator << (std::ostream& out, const OneD<_TT>& p);
    OneD<_T>& operator *= (double cv);
    OneD<_T> operator * (double cv) const;
    template<typename _TT>
        friend OneD<_TT> operator * (double cv, const OneD<_TT>& p);
    OneD<_T>& operator *= (const OneD<_T>& p);
    OneD<_T> operator * (const OneD<_T>& p) const;
    OneD<_T>& operator += (const OneD<_T>& p);
    OneD<_T> operator + (const OneD<_T>& p) const;
    OneD<_T>& operator -= (const OneD<_T>& p);
    OneD<_T> operator - (const OneD<_T>& p) const;
    OneD<_T>& operator /= (double cv);
    OneD<_T> operator / (double cv) const;

    template<typename Vector_T> bool operator==(const Vector_T& p) const
    {
	size_t sz = size();
	size_t psz = p.size();
	unsigned int minSize = std::min<unsigned int>(sz,
						      psz);
	
	for (unsigned int i = 0 ; i < minSize ; i++)
	    if (!math::linear::equals(mCoef[i], p[i]))
		return false;
	
	_T dflt(0.0);
	
	// Cover case where one polynomial has more 
	// coefficients than the other.
	if (sz > psz)
	{
	    for (unsigned int i = minSize; i < sz; i++)
		if (!math::linear::equals(mCoef[i], dflt))
		    return false;
	}
	else if (sz < psz)
	{
	    for (unsigned int i = minSize; i < psz; i++)
		if (!math::linear::equals(p[i], dflt))
		    return false;
	}
	
	return true;
    }

    
    template<typename Vector_T> bool operator!=(const Vector_T& p) const
    {
	return !(*this == p);
    }
};

/*!
 *  Templated function to perform a linear least squares fit for the data.
 *  This algorithm is fairly straightforward.
 *
 *  To fit an order N polynomial, we need to solve
 *  Ax=b, for x.
 *
 *  A is a system of polynomials, e.g.
 *
 *  f(x) = c0 + c1*x + c2*x^2 + c3*x^3 = y
 *  
 *  Each observed point in the data sets is computed
 *  for our A matrix.
 *
 *  e.g.: f(1) = 3, f(-1) = 13, f(2) 1, f(-2) = 33
 *
 *  | 1  1  1  1 || c0 |   |  3 |
 *  | 1 -1  1 -1 || c1 | = | 13 |
 *  | 1  2  4  8 || c2 |   |  1 |
 *  | 1 -2  4 -8 || c3 |   | 33 |
 *
 *  
 *  Linear least squares solution for system where
 *  ker(A) = {0} (IOW, there are free variables)
 *
 *  x = inv(A' * A) * A' * b
 *
 *  \param x The observable x points
 *  \param y The observable y solutions
 *  \return A one dimensional polynomial that fits the curve
 */
 
template<typename Vector_T> OneD<double> fit(const Vector_T& x,
					     const Vector_T& y,
					     int numCoeffs);

/*!
 *  This method allows us to fit a set of observations using raw
 *  pointers
 */
inline OneD<double> fit(int numObs, const double* x, const double* y, 
			int numCoeffs)
{
    math::linear::Vector<double> xv(numObs, x);
    math::linear::Vector<double> yv(numObs, y);
    return math::poly::fit<math::linear::Vector<double> >(xv, yv, numCoeffs);
}


} // poly
} // math
#include "math/poly/OneD.hpp"
#endif
