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

#ifndef __LANG_STL_LIST_H__
#define __LANG_STL_LIST_H__

#include "lang/List.h"
#include "lang/Defaults.h"
#include <list>

namespace lang    
{

template<typename T>
class STLListIterator: public Iterator<T>
{
public:
    typedef typename std::list<T>::const_iterator Iter;

    STLListIterator(const Iter iter, const Iter end) :
        mIter(iter), mEnd(end)
    {
    }

    virtual ~STLListIterator()
    {
    }

    bool hasNext() const
    {
        return mIter != mEnd;
    }

    T next()
    {
        if (!hasNext())
            throw except::NullPointerReference(Ctxt("No elements left"));
        T t = *mIter;
        mIter++;
        return t;
    }

protected:
    Iter mIter, mEnd;
};


template<typename Element_T,
         typename Comparator_T = DefaultComparator<Element_T> ,
         typename Cloner_T = DefaultCloner<Element_T>,
         typename Destructor_T = DefaultDestructor<Element_T> >
class STLList: public List<Element_T>
{

public:
    STLList()
    {
    }

    virtual ~STLList()
    {
        destroy();
    }

    typedef STLList<Element_T, Comparator_T, Cloner_T, Destructor_T> List_T;
    typedef std::auto_ptr< ::lang::Iterator<Element_T> > Iterator;

    virtual Iterator iterator()
    {
        return iterator(0);
    }
    virtual Iterator iterator() const
    {
        return iterator(0);
    }

    virtual Iterator iterator(int index, int span = SLICE_MAX) const
    {
        size_t elems = size();
        //TODO currently, negative indexing is not allowed
        //I would like to change that
        if (index < 0 || (index >= elems && elems > 0) || span < 0)
            throw except::IndexOutOfRangeException(Ctxt(
                    "negative slices are not supported yet"));

        span = std::min<int>(span, elems - index);
        int revNum = std::max<int>(elems - span - index, 0);
        ConstIter_T iter = mList.begin();
        ConstIter_T end = mList.end();

        for(int i = 0; i < index; ++i)
            iter++;
        for(int i = 0; i < revNum; ++i)
            end--;
        return Iterator(new STLListIterator<Element_T> (iter, end));
    }

    virtual size_t size() const
    {
        return mList.size();
    }

    virtual Element_T& front() throw (except::IndexOutOfRangeException)
    {
        if (size() <= 0)
            throw except::IndexOutOfRangeException(Ctxt("No elements"));
        return mList.front();
    }
    virtual Element_T& back() throw (except::IndexOutOfRangeException)
    {
        if (size() <= 0)
            throw except::IndexOutOfRangeException(Ctxt("No elements"));
        return mList.back();
    }

    virtual Element_T& operator[](size_t index)
                throw (except::IndexOutOfRangeException)
    {
        size_t elems = size();
        if (index >= elems)
            throw except::IndexOutOfRangeException(Ctxt(FmtX(
                    "Index out of range: (%d <= %d < %d)", 0, index, elems)));
        Iter_T iter = mList.begin();
        for (size_t count = 0; count < index; ++count, ++iter);
        return *iter;
    }

    virtual Element_T operator[](size_t index) const
            throw (except::IndexOutOfRangeException)
    {
        return Parent_T::get(index);
    }

    virtual void insert(const Element_T& element, size_t index = INSERT_END)
    {
        if (index >= size())
        {
            mList.push_back(element);
        }
        else
        {
            Iter_T iter = mList.begin();
            Iter_T end = mList.end();
            for(size_t i = 0; i < index && iter != end; ++i)
            {
                iter++;
            }
            mList.insert(iter, element);
        }
    }

    virtual bool remove(const Element_T& element)
    {
        Iter_T it = mList.begin();
        Iter_T end = mList.end();
        for(; it != end; ++it)
        {
            Element_T& e = *it;
            if (mComparator(e, element))
            {
                mList.erase(it);
                return true;
            }
        }
        return false;
    }

    virtual bool contains(const Element_T& element) const
    {
        try
        {
            indexOf(element);
            return true;
        }
        catch (except::NoSuchReferenceException& ex)
        {
            return false;
        }
    }

    virtual void clear()
    {
        destroy();
    }

    virtual List_T* clone() const
    {
        TrueFilter<Element_T> identity;
        return filter(identity, true);
    }

    virtual List_T* filter(Filter<Element_T>& f, bool clone = true) const
    {
        List_T* filtered = new List_T;
        Iterator it = iterator();
        while (it->hasNext())
        {
            Element_T elem = it->next();
            if (f(elem))
            {
                if (clone)
                    filtered->add(mCloner(elem));
                else
                    filtered->add(elem);
            }
        }
        return filtered;
    }

protected:
    typedef List<Element_T> Parent_T;
    typedef std::list<Element_T> Storage_T;
    typedef typename Storage_T::iterator Iter_T;
    typedef typename Storage_T::const_iterator ConstIter_T;
    Storage_T mList;
    Comparator_T mComparator;
    Cloner_T mCloner;
    Destructor_T mDestructor;

    void destroy()
    {
        Iter_T iter = mList.begin();
        Iter_T end = mList.end();
        for(; iter != end; ++iter)
        {
            mDestructor(*iter);
        }
        mList.clear();
    }

    virtual bool compareElements(const Element_T& e1, const Element_T& e2) const
    {
        return mComparator(e1, e2);
    }
    virtual Element_T cloneElement(const Element_T& e) const
    {
        return mCloner(e);
    }
};

}
#endif
