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

#ifndef __LANG_COLLECTION_H__
#define __LANG_COLLECTION_H__

#include <import/except.h>
#include "lang/Iterator.h"
#include "lang/Filter.h"
#include <memory>

namespace lang    
{

/**
 * @brief Collection Interface
 */
template<typename Element_T>
class Collection
{

public:
    typedef Collection<Element_T> Collection_T;

    Collection()
    {
    }

    virtual ~Collection()
    {
    }

    typedef std::auto_ptr< ::lang::Iterator<Element_T> > Iterator;

    /**
     * @return a ConstIterator for the elements in the Collection
     */
    virtual Iterator iterator() const = 0;

    /**
     * @return the size of the Dictionary (number of elements)
     */
    virtual size_t size() const = 0;

    /**
     * @return the size of the Collection (number of elements)
     */
    bool empty() const
    {
        return size() <= 0;
    }

    /**
     * Adds the given element to the collection.
     */
    virtual void add(const Element_T& element) = 0;

    virtual void addAll(Iterator& it, bool clone = true) = 0;

    /**
     * Removes the object from the collection, if it exists.
     * This does not destroy the object since the caller already has a handle
     * to it.
     * @return whether the element was removed (i.e. if it existed)
     */
    virtual bool remove(const Element_T& element) = 0;

    /**
     * @return -
     */
    virtual bool contains(const Element_T& element) const = 0;

    virtual void clear() = 0;

    /**
     * Clones the Dictionary
     * @return a deep copy of the Dictionary
     */
    virtual Collection_T* clone() const = 0;

    /**
     * Filters the dictionary with the given Filter
     * @return a new Dictionary with the filtered elements
     */
    virtual Collection_T
            * filter(Filter<Element_T>& filter, bool clone = true) const = 0;

};

}
#endif
