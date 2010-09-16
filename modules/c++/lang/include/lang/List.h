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

#ifndef __LANG_LIST_H__
#define __LANG_LIST_H__

#include <import/except.h>
#include "lang/Iterator.h"
#include "lang/Filter.h"
#include "lang/Collection.h"
#include <memory>
#include <limits>

#undef min
#undef max

namespace lang    
{

    const int SLICE_MAX = std::numeric_limits<int>::max();
    const size_t INSERT_END = std::numeric_limits<size_t>::max();

/**
 * @brief List Interface
 */
template<typename Element_T>
class List: public Collection<Element_T>
{

public:
    List()
    {
    }

    virtual ~List()
    {
    }

    typedef List<Element_T> List_T;
    typedef std::auto_ptr< ::lang::Iterator<Element_T> > Iterator;

    virtual Iterator iterator() const = 0;

    virtual Iterator iterator(int index, int span = SLICE_MAX) const = 0;

    virtual size_t size() const = 0;

    virtual void add(const Element_T& element)
    {
        insert(element);
    }

    virtual void append(const Element_T& element)
    {
        insert(element);
    }

    virtual void insert(const Element_T& element,
                        size_t index = INSERT_END) = 0;

    virtual Element_T operator[](size_t index) const
            throw (except::IndexOutOfRangeException)
    {
        return get(index);
    }

    virtual Element_T& front() throw (except::IndexOutOfRangeException) = 0;
    virtual Element_T& back() throw (except::IndexOutOfRangeException) = 0;
    virtual Element_T& operator[](size_t index)
            throw (except::IndexOutOfRangeException) = 0;

    virtual Element_T get(size_t index) const
            throw (except::IndexOutOfRangeException)
    {
        size_t count = 0;
        Iterator it = iterator();
        while(it->hasNext() && count < index)
        {
            count++;
            it->next();
        }
        if (count == index && it->hasNext())
            return it->next();
        throw except::IndexOutOfRangeException(Ctxt(FmtX(
                "Index out of range: (%d <= %d < %d)", 0, index, count)));
    }

    virtual bool remove(const Element_T& element) = 0;

    virtual bool contains(const Element_T& element) const = 0;

    virtual void clear() = 0;

    virtual void addAll(Iterator& it, bool clone = true)
    {
        while (it->hasNext())
        {
            Element_T e = it->next();
            if (clone)
                add(cloneElement(e));
            else
                add(e);
        }
    }

    virtual size_t indexOf(const Element_T& element) const
            throw (except::NoSuchReferenceException)
    {
        size_t index = 0;
        Iterator it = iterator();
        while (it->hasNext())
        {
            Element_T e = it->next();
            if (compareElements(e, element))
                return index;
            index++;
        }
        throw except::NoSuchReferenceException(Ctxt("element not found"));
    }

    virtual size_t lastIndexOf(const Element_T& element) const
            throw (except::NoSuchReferenceException)
    {
        size_t index = 0, foundIndex = 0;
        bool found = false;
        Iterator it = iterator();
        //maybe not the best way of doing this - which is why it is virtual
        while (it->hasNext())
        {
            Element_T e = it->next();
            if (compareElements(e, element))
            {
                found = true;
                foundIndex = index;
            }
            index++;
        }
        if (found)
            return foundIndex;
        throw except::NoSuchReferenceException(Ctxt("element not found"));
    }

    virtual List_T* clone() const = 0;

    virtual List_T* filter(Filter<Element_T>& f, bool clone = true) const = 0;

protected:
    virtual bool compareElements(const Element_T& e1, const Element_T& e2) const = 0;
    virtual Element_T cloneElement(const Element_T& e) const = 0;
};

}
#endif
