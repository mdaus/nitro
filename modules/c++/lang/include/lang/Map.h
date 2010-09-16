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

#ifndef __LANG_MAP_H__
#define __LANG_MAP_H__

#include "lang/Iterator.h"
#include "lang/Filter.h"
#include "lang/Utility.h"
#include <import/except.h>
#include <import/sys.h>
#include <memory>

namespace lang    
{

/**
 * @brief Generic Map Interface
 */
template<typename Key_T, typename Value_T>
class Map
{
public:
    Map()
    {
    }
    virtual ~Map()
    {
    }

    typedef ::lang::Pair<Key_T, Value_T> Pair;
    typedef Map<Key_T, Value_T> Map_T;
    typedef std::auto_ptr< ::lang::Iterator<Pair> > Iterator;

    /**
     * @return an Iterator
     */
    virtual Iterator iterator() const = 0;

    /**
     * @return true if the key exists, otherwise false
     */
    virtual bool exists(Key_T& key) const = 0;

    /**
     * @return the size of the Map (number of elements)
     */
    virtual size_t size() const = 0;

    /**
     * @return true if the dictionary is empty
     */
    bool empty() const
    {
        return size() <= 0;
    }

    /**
     * Removes object with the given Key_T from the dictionary and returns it.
     * @return the object
     * @throw except::NoSuchKeyException if the object does not exist
     */
    virtual Value_T pop(const Key_T& key) throw (except::NoSuchKeyException) = 0;

    /**
     * Removes object with the given Key_T from the dictionary. If the object
     * does not exist, no exception is thrown.
     */
    virtual bool remove(const Key_T& key) = 0;

    /**
     * @return the Value_T that has the given Key_T
     * @throw except::NoSuchKeyException if the object does not exist
     */
    virtual Value_T operator[](const Key_T& key) const
            throw (except::NoSuchKeyException) = 0;

    /**
     * Sets the object with the given Key_T
     * @return a reference to the object
     */
    virtual Value_T& operator[](const Key_T& key) = 0;

    /**
     * @return the Value_T that has the given Key_T
     * @throw except::NoSuchKeyException if the object does not exist
     */
    Value_T get(const Key_T& key) const throw (except::NoSuchKeyException)
    {
        return (*this)[key];
    }

    /**
     * Inserts the Value_T with the given Key_T into the Map
     * @return the value
     */
    Value_T put(const Key_T& key, Value_T value)
    {
        (*this)[key] = value;
        return value;
    }

    /**
     * Clears the dictionary (removes all elements)
     */
    virtual void clear() = 0;

    /**
     * Clones the Map
     * @return a deep copy of the Map
     */
    virtual Map_T* clone() const = 0;

    /**
     * Filters the dictionary with the given Filter
     * @return a new Map with the filtered elements
     */
    virtual Map_T* filter(Filter<Pair>& filter) const = 0;
};

}

#endif
