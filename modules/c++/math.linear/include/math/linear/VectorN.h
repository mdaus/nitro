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
#ifndef __MATH_LINEAR_VECTOR_N_H__
#define __MATH_LINEAR_VECTOR_N_H__

#include "math/linear/VectorN.h"

namespace math
{
namespace linear
{
 
template<size_t _ND, typename _T=double> class VectorN
{
    typedef VectorN<_ND, _T> Like_T;
    MatrixMxN<_ND, 1, _T> mRaw;
    
public:



    VectorN() 
    {
        VectorN(0.0);
    }

    VectorN(_T sv)
    {
        mRaw = sv;
    }

    ~VectorN()
    {

    }

    VectorN(const VectorN& v)
    {
        mRaw = v.mRaw;
    }
    VectorN& operator=(const VectorN& v)
    {
        if (this != &v)
        {
            mRaw = v.mRaw;
        }
        return *this;
    }

    VectorN& operator=(const _T& sv)
    {
        mRaw = sv;
        return *this;
    }

    VectorN(const MatrixMxN<_ND, 1, _T>& mx)
    {
        mRaw = mx;
    }

    MatrixMxN<_ND, 1, _T>& matrix() { return mRaw; }
    const MatrixMxN<_ND, 1, _T>& matrix() const { return mRaw; }

    inline _T operator[](int i) const
    {
#if defined(MATH_LINEAR_BOUNDS)
        assert( i < _ND );
#endif
        return mRaw[i][0];
    }
    inline _T& operator[](int i)
    {
#if defined(MATH_LINEAR_BOUNDS)
        assert( i < _ND );
#endif
        return mRaw[i][0];

    }

    _T dot(const VectorN<_ND>& vec) const
    {
        _T acc(0);
        for (unsigned int i = 0; i < _ND; ++i)
        {
            acc += (*this)[i] * vec[i];
        }
        return acc;
    }

    /*!
     * Euclidean, L2 norm
     */
    _T norm() const
    {
        _T acc(0);
        for (unsigned int i = 0; i < _ND; ++i)
        {
            acc += (*this)[i] * (*this)[i];
        }
        return (_T)::sqrt((const _T)acc);
    }

    void normalize()
    {
        mRaw.scale(1.0/norm());
    }

    void scale(_T scalar)
    {
        mRaw.scale(scalar);
    }

    Like_T& 
    operator+=(const Like_T& v)
    {
        mRaw += v.matrix();
        return *this;
    }
    Like_T&
    operator-=(const Like_T& v)
    {
        mRaw -= v.matrix();
        return *this;
    }

    Like_T add(const Like_T& v) const
    {
        Like_T v2(*this);
        v2 += v;
        return v2;
    }

    Like_T subtract(const Like_T& v) const
    {
        Like_T v2(*this);
        v2 -= v;
        return v2;
    }

    Like_T 
    operator+(const Like_T& v) const
    {
        return add(v);
    }
    Like_T
    operator-(const Like_T& v) const
    {
        return subtract(v);
    }

    Like_T& operator *=(const Like_T& v)
    {
        for (unsigned int i = 0; i < _ND; i++)
        {
            mRaw(i, 0) *= v[i];
        }
        return *this;
        
    }

    Like_T& operator *=(_T sv)
    {
        scale(sv);
        return *this;
        
    }

    Like_T operator *(_T sv) const
    {
        
        Like_T v2(*this);
        v2 *= sv;
        return v2;
        
    }


    Like_T& operator /=(const Like_T& v)
    {
        for (unsigned int i = 0; i < _ND; i++)
        {
            mRaw(i, 0) /= v[i];
        }
        return *this;
    }

    Like_T operator*(const Like_T& v) const
    {
        Like_T v2(*this);
        v2 *= v;
        return v2;
    }

    Like_T operator/(const Like_T& v) const
    {
        Like_T v2(*this);
        v2 /= v;
        return v2;
    }

};


template<typename _T> VectorN<3, _T> cross(const VectorN<3, _T>& u,
                                           const VectorN<3, _T>& v)
{
    VectorN<3, _T> xp;
    xp[0] = (u[1]*v[2] - u[2]*v[1]);
    xp[1] = (u[2]*v[0] - u[0]*v[2]);
    xp[2] = (u[0]*v[1] - u[1]*v[0]);
    return xp;
}

template<size_t _ND, typename _T> VectorN<_ND, _T> 
    constantVector(_T cv = 0)
{
    VectorN<_ND, _T> v(math::linear::constantMatrix<_ND, 1, _T>(cv));
    return v;
}

}
}

template<size_t _MD, size_t _ND, typename _T> 
    math::linear::VectorN<_MD, _T>
    operator*(const math::linear::MatrixMxN<_MD, _ND, _T>& m, 
              const math::linear::VectorN<_ND, _T>& v)
{
    return math::linear::VectorN<_MD, _T>(m * v.matrix());
}

template<size_t _ND, typename _T> math::linear::VectorN<_ND, _T>
    operator*(_T scalar, const math::linear::VectorN<_ND, _T>& v)
{
    return v * scalar;
}


template<size_t _ND, typename _T> 
    std::ostream& operator<<(std::ostream& os,
                             const math::linear::VectorN<_ND, _T>& v)
{
    for (unsigned int i = 0; i < _ND; ++i)
    {
        os << v[i] << std::endl;
    }
    return os;
    
}

#endif
