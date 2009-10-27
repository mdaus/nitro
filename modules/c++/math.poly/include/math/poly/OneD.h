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

namespace math
{
namespace poly
{
/*! This class defines an implementation for a 1-D polynomial. It should
    support datatypes that allow the following operators:
    _T(0), _T*_T, _T*=_T, _T+_T, _T+=_T, _T-_T, _T-=_T, -T*<double>

    It supports polynomials of the form:
    a0 + a1*x + a2*x^2 + ...

    It supports evaluating the integral over a specified interval (a...b)

    It supports computing the derivative

    And, it supports the multiplication/addtion/subtraction of 1-D polynomials.
*/
template<typename _T>
class OneD
{
protected:
    std::vector<_T> mCoef;
    
public:
    OneD() {}
    OneD(int order) : mCoef(order+1,0) {}
    OneD(const std::vector<_T>& coef) : mCoef(coef) {}
    template<typename Cont_T> OneD(const Cont_T& coeff) 
    {
        size_t sizeC = coeff.size();
        mCoef.resize(sizeC);
        for (unsigned int i = 0; i < sizeC; i++)
        {
            mCoef[i] = coeff[i];
        }
    }
    int order() const { return mCoef.size()-1; }
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
};

} // poly
} // math
#include "math/poly/OneD.hpp"
#endif
