/* =========================================================================
 * This file is part of lang-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2010, General Dynamics - Advanced Information Systems
 *
 * logging-c++ is free software; you can redistribute it and/or modify
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

#ifndef __LANG_UTILITY_H__
#define __LANG_UTILITY_H__

#include <import/except.h>
#include <import/sys.h>

namespace lang    
{

template<typename First_T, typename Second_T>
struct Pair
{
    Pair()
    {
    }
    Pair(First_T _first, Second_T _second) :
        first(_first), second(_second)
    {
    }
    ~Pair()
    {
    }
    First_T first;
    Second_T second;
};

template<typename Element_T, size_t Size_T>
class Tuple
{
public:
    Tuple()
    {
    }
    ~Tuple()
    {
    }

    Element_T operator[](size_t index) const
            throw (except::IndexOutOfRangeException)
    {
        if (index < 0 || index >= Size_T)
            throw except::IndexOutOfRangeException(Ctxt(FmtX(
                    "Index out of range: (%d <= %d < %d)", 0, index, Size_T)));
        return mElems[index];
    }

    Element_T& operator[](size_t index)
            throw (except::IndexOutOfRangeException)
    {
        if (index < 0 || index >= Size_T)
            throw except::IndexOutOfRangeException(Ctxt(FmtX(
                    "Index out of range: (%d <= %d < %d)", 0, index, Size_T)));
        return mElems[index];
    }

    size_t size() const
    {
        return Size_T;
    }

protected:
    Element_T mElems[Size_T];
};

}
#endif
