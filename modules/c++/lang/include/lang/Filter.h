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

#ifndef __LANG_FILTER_H__
#define __LANG_FILTER_H__

namespace lang
{

template<typename T>
class Filter
{
public:
    Filter()
    {
    }
    virtual ~Filter()
    {
    }
    virtual bool operator()(const T& obj) = 0;
};

/**
 * Always returns true - good for identity filtering/copying
 */
template<typename T>
class TrueFilter: public Filter<T>
{
public:
    TrueFilter()
    {
    }
    virtual ~TrueFilter()
    {
    }
    bool operator()(const T& obj)
    {
        return true;
    }
};

}
#endif
