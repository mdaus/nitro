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

#include <import/except.h>
#include <import/sys.h>
#include "math/poly/OneD.h"

namespace math
{
namespace poly
{

template<typename _T>
_T 
TwoD<_T>::operator () (double atX, double atY) const
{
    _T lRet(0.0);
    double lAtXPwr = 1.0;
    for (unsigned int lX = 0, lXEnd = mCoef.size() ; lX < lXEnd ; lX++)
    {
        lRet += mCoef[lX](atY)*lAtXPwr;
        lAtXPwr *= atX;
    }
    return lRet;
}
    
template<typename _T>
_T 
TwoD<_T>::integrate(double xStart, double xEnd, double yStart, double yEnd) const
{
    _T lRet(0.0);
    double lEndAtPwr = xEnd;
    double lStartAtPwr = xStart;
    double lDiv = 0;
    double lNewCoef;
    for (unsigned int lX = 0, lXEnd = mCoef.size() ; lX < lXEnd ; lX++)
    {
        lDiv = 1.0 / (lX + 1);
        lNewCoef = mCoef[lX].integrate(yStart,yEnd) * lDiv;
        lRet += lNewCoef * lEndAtPwr;
        lRet -= lNewCoef * lStartAtPwr;
        lEndAtPwr *= xEnd;
        lStartAtPwr *= xStart;
    }
    return lRet;
}
    
template<typename _T>
TwoD<_T> 
TwoD<_T>::derivativeY() const
{
    TwoD<_T> lRet;
    if ((orderY() > 0))
    {
        lRet = TwoD<_T>(orderX(), orderY()-1);
        for (unsigned int lX = 0 ; lX < mCoef.size() ; lX++)
        {
            lRet.mCoef[lX] = mCoef[lX].derivative();
        }
    }
    return lRet;
}

template<typename _T>
TwoD<_T> 
TwoD<_T>::derivativeX() const
{
    TwoD<_T> lRet;
    if ((orderX() > 0))
    {
        lRet = TwoD<_T>(orderX()-1, orderY());
        for (unsigned int lX = 0 ; lX < mCoef.size()-1 ; lX++)
        {
            lRet.mCoef[lX] = mCoef[lX+1] * (_T)(lX+1);
        }
    }
    return lRet;
}

template<typename _T>
OneD<_T>
TwoD<_T>::atY(double y) const
{
    OneD<_T> lRet;
    if (orderX() > 0)
    {
        // We have no more X, but we have Y still
        lRet = OneD<_T>(orderX());
        for (unsigned int lX = 0; lX < mCoef.size(); lX++)
        {
            // Get me down to an X
            lRet[lX] = mCoef[lX](y);
        }
    }
    return lRet;
}

template<typename _T>
TwoD<_T>
TwoD<_T>::power(int toThe) const
{
    // If its 0, we have to give back a 1*x^0*y^0 poly, since
    // we want a 2D poly out
    if (toThe == 0)
    {
        TwoD<_T> zero(0, 0);
        zero[0][0] = 1;
        return zero;
    }

    TwoD<double> rv = *this;

    // If its 1 give it back now
    if (toThe == 1)
        return rv;

 
    // Otherwise, we have to raise it
    
    for (int i = 2; i <= toThe; i++)
    {
        rv *= *this;
    }
    return rv;

}


template<typename _T>
TwoD<_T> 
TwoD<_T>::flipXY() const
{
    unsigned int oY = orderX();
    unsigned int oX = orderY();
    TwoD<_T> prime(oX, oY);
    
    for (unsigned int i = 0; i <= oX; i++)
	for (unsigned int j = 0; j <= oY; j++)
	    prime[i][j] = mCoef[j][i];
    return prime;
}

template<typename _T>
TwoD<_T> 
TwoD<_T>::derivativeXY() const
{
    TwoD<_T> lRet = derivativeY().derivativeX();
    return lRet;
}
    
template<typename _T>
OneD<_T> 
TwoD<_T>::operator [] (unsigned int idx) const
{
    OneD<_T> lRet(0.0);
    if (idx < mCoef.size())
    {
        lRet = mCoef[idx];
    }
    else
    {
        std::stringstream lStr;
        lStr << "idx(" << idx << ") not within range [0..." << mCoef.size() << ")";
        std::string lMsg(lStr.str());
        throw(except::IndexOutOfRangeException(Ctxt(lMsg)));
    }
    return lRet;
}

template<typename _T>
_T* 
TwoD<_T>::operator [] (unsigned int idx) 
{
    if (idx < mCoef.size())
    {
        return(&(mCoef[idx][0]));
    }
    else
    {
        std::stringstream lStr;
        lStr << "idx(" << idx << ") not within range [0..." << mCoef.size() << ")";
        std::string lMsg(lStr.str());
        throw(except::IndexOutOfRangeException(Ctxt(lMsg)));
    }
}
    
template<typename _T>
TwoD<_T>& 
TwoD<_T>::operator *= (double cv) 
{
    for (unsigned int lX = 0, lXEnd = mCoef.size() ; lX < lXEnd ; lX++)
    {
        mCoef[lX] *= cv;
    }
    return *this;
}
    
template<typename _T>
TwoD<_T> 
TwoD<_T>::operator * (double cv) const
{
    TwoD<_T> lRet(*this);
    lRet *= cv;
    return lRet;
}

template<typename _T>
TwoD<_T> 
operator * (double cv, const TwoD<_T>& p) 
{
    return p*cv;
}
    
template<typename _T>
TwoD<_T>& 
TwoD<_T>::operator *= (const TwoD<_T>& p) 
{
    TwoD<_T> lTmp(orderX()+p.orderX(),orderY()+p.orderY());
    for (int lX = 0, lXEnd = mCoef.size() ; lX < lXEnd ; lX++)
    {
        for (int lY = 0, lYEnd = p.mCoef.size() ; lY < lYEnd ; lY++)
        {
            lTmp.mCoef[lX+lY] += mCoef[lX] * p.mCoef[lY];
        }
    }
    *this = lTmp;
    return *this;
}

template<typename _T>
TwoD<_T> 
TwoD<_T>::operator * (const TwoD<_T>& p) const
{
    TwoD<_T> lRet(*this);
    lRet *= p;
    return lRet;
}

template<typename _T>
TwoD<_T>& 
TwoD<_T>::operator += (const TwoD<_T>& p) 
{
    TwoD<_T> lTmp(std::max<int>(orderX(),p.orderX()),
                  std::max<int>(orderY(),p.orderY()));
    for (unsigned int lX = 0, lXEnd = mCoef.size() ; lX < lXEnd ; lX++)
    {
        lTmp.mCoef[lX] = mCoef[lX];
    }
    for (unsigned int lX = 0, lXEnd = p.mCoef.size() ; lX < lXEnd ; lX++)
    {
        lTmp.mCoef[lX] += p.mCoef[lX];
    }
    *this = lTmp;
    return *this;
}
    
template<typename _T>
TwoD<_T> 
TwoD<_T>::operator + (const TwoD<_T>& p) const
{
    TwoD<_T> lRet(*this);
    lRet += p;
    return lRet;
}

template<typename _T>
TwoD<_T>& 
TwoD<_T>::operator -= (const TwoD<_T>& p) 
{
    TwoD<_T> lTmp(std::max<int>(orderX(),p.orderX()),
                  std::max<int>(orderY(),p.orderY()));
    for (unsigned int lX = 0, lXEnd = mCoef.size() ; lX < lXEnd ; lX++)
    {
        lTmp.mCoef[lX] = mCoef[lX];
    }
    for (unsigned int lX = 0, lXEnd = p.mCoef.size() ; lX < lXEnd ; lX++)
    {
        lTmp.mCoef[lX] -= p.mCoef[lX];
    }
    *this = lTmp;
    return *this;
}
    
template<typename _T>
TwoD<_T> 
TwoD<_T>::operator - (const TwoD<_T>& p) const
{
    TwoD<_T> lRet(*this);
    lRet -= p;
    return lRet;
}
    
template<typename _T>
TwoD<_T>& 
TwoD<_T>::operator /= (double cv) 
{
    double lRecipCV = 1.0/cv;
    for (unsigned int lX = 0, lXEnd = mCoef.size() ; lX < lXEnd ; lX++)
    {
        mCoef[lX] *= lRecipCV;
    }
    return *this;
}

template<typename _T>
TwoD<_T> 
TwoD<_T>::operator / (double cv) const
{
    TwoD<_T> lRet(*this);
    lRet *= (1.0/cv);
    return lRet;
}

template<typename _T>
std::ostream& 
operator << (std::ostream& out, const TwoD<_T> p)
{
    for (unsigned int lX = 0 ; lX < p.mCoef.size() ; lX++)
    {
        out << "x^" << lX << "*(" << p[lX] << ")" << std::endl;
    }
    return out;
}

template<typename _T>
bool
TwoD<_T>::operator == (const TwoD<_T>& p) const
{
    unsigned int lMinSize = std::min<unsigned int>(mCoef.size(),
            p.mCoef.size());

    for (unsigned int lX = 0 ; lX < lMinSize ; lX++)
    {
        if (mCoef[lX] != p.mCoef[lX])
            return false;
    }

    // Cover case where one polynomial has more coefficients than the other.
    if (mCoef.size() > p.mCoef.size())
    {
        OneD<_T> lDflt(orderY());

        for (unsigned int lX = lMinSize ; lX < mCoef.size() ; ++lX)
            if (mCoef[lX] != lDflt)
                return false;
    }
    else if (mCoef.size() < p.mCoef.size())
    {
        OneD<_T> lDflt(p.orderY());

        for (unsigned int lX = lMinSize ; lX < p.mCoef.size() ; ++lX)
            if (p.mCoef[lX] != lDflt)
                return false;
    }

    return true;
}

template<typename _T>
bool
TwoD<_T>::operator != (const TwoD<_T>& p) const
{
    return !(*this == p);
}

} // poly
} // math
