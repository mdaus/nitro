/* =========================================================================
 * This file is part of lang-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2010, General Dynamics - Advanced Information Systems
 *
 * lang-c++ is free software; you can redistribute it and/or modify
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

#ifndef __LANG_DEFAULTS_H__
#define __LANG_DEFAULTS_H__

#include <import/except.h>
#include <import/str.h>

namespace lang    
{

/**
 * Default Comparator assumes the type supports the == operator.
 */
template <typename T>
class DefaultComparator
{
public:
    ~DefaultComparator(){}
    virtual bool operator() (const T& obj1, const T& obj2) const
    {
        return obj1 == obj2;
    }
};


/**
 * DefaultCloner doesn't actually clone. It assumes the templated type is not
 * a pointer and that the copy constructor is used.
 */
template <typename T>
class DefaultCloner
{
public:
    ~DefaultCloner(){}
    virtual T operator() (const T& obj) const
    {
        return obj;
    }
};

/**
 * PointerCloner assumes a pointer template that supports the clone() method.
 */
template <typename T>
class PointerCloner
{
public:
    ~PointerCloner(){}
    virtual T operator() (const T& obj) const
    {
        return obj->clone();
    }
};

/**
 * DefaultDestructor does nothing.
 */
template <typename T>
class DefaultDestructor
{
public:
    ~DefaultDestructor(){}
    virtual void operator() (T& obj)
    {
        //does nothing
    }
};

/**
 * DeleteDestructor assumes a pointer template and calls delete on the object.
 */
template <typename T>
class DeleteDestructor
{
public:
    ~DeleteDestructor(){}
    virtual void operator() (T& obj)
    {
        delete obj;
    }
};

/**
 * Default hash function. Turns the key into a string and hashes that.
 */
template <typename T = std::string>
class DefaultStringHash
{
public:
    ~DefaultStringHash(){}
    virtual size_t operator() (const T& obj) const
    {
        std::string k = str::toString(obj);
        char c;
        size_t hash = 0;

        for(size_t i = 0, len = k.size(); i < len; ++i)
        {
            char c = k[i];
            if (c > 0140)
                c -= 40;
            hash = ((hash << 3) + (hash >> 28) + c);
        }
        return (size_t) ((hash & 07777777777));
    }
};

}
#endif
