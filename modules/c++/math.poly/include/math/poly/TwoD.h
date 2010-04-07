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
    TwoD(int orderX, int orderY) : mCoef(orderX+1,OneD<_T>(orderY)) {}; 
    int orderX() const { return mCoef.size()-1; };
    int orderY() const { return (orderX() < 0 ? -1 : mCoef[0].order()); };
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

} // poly
} // math
#include "math/poly/TwoD.hpp"
#endif
