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

#ifndef __LANG_STL_MAP_H__
#define __LANG_STL_MAP_H__

#include "lang/Map.h"
#include "lang/Defaults.h"
#include "lang/Utility.h"
#include <map>

namespace lang    
{

template<typename K, typename V>
class STLMapIterator: public Iterator<Pair<K, V> >
{
public:
    typedef typename std::map<K, V>::const_iterator Iter;
    typedef ::lang::Pair<K, V> Pair;

    STLMapIterator(const Iter iter, const Iter end) :
        mIter(iter), mEnd(end)
    {
    }

    virtual ~STLMapIterator()
    {
    }

    bool hasNext() const
    {
        return mIter != mEnd;
    }

    Pair next()
    {
        if (!hasNext())
            throw except::NullPointerReference(Ctxt("No elements left"));
        const std::pair<const K, V> p = *mIter;
        mIter++;
        return Pair(p.first, p.second);
    }

protected:
    Iter mIter, mEnd;
};

/**
 * @brief std::map implementation of the Map
 */
template<typename Key_T,
         typename Value_T,
         typename Cloner_T = DefaultCloner<Value_T>,
         typename Destructor_T = DefaultDestructor<Value_T> >
class STLMap: public Map<Key_T, Value_T>
{
public:
    STLMap()
    {
    }
    virtual ~STLMap()
    {
        destroy();
    }

    typedef ::lang::Pair<Key_T, Value_T> Pair;
    typedef STLMap<Key_T, Value_T, Cloner_T, Destructor_T> Map_T;
    typedef std::auto_ptr< ::lang::Iterator<Pair> > Iterator;

    virtual Iterator iterator() const
    {
        return Iterator(
                new STLMapIterator<Key_T, Value_T> (mMap.begin(), mMap.end()));
    }

    virtual bool exists(Key_T& key) const
    {
        return mMap.find(key) != mMap.end();
    }

    virtual size_t size() const
    {
        return mMap.size();
    }

    virtual Value_T operator[](const Key_T& key) const
            throw (except::NoSuchKeyException)
    {
        ConstIter_T p = mMap.find(key);
        if (p == mMap.end())
            throw except::NoSuchKeyException(Ctxt(str::toString(key)));
        return p->second;
    }

    virtual Value_T pop(const Key_T& key) throw (except::NoSuchKeyException)
    {
        Iter_T p = mMap.find(key);
        if (p == mMap.end())
            throw except::NoSuchKeyException(Ctxt(str::toString(key)));
        Value_T& value = p->second;
        mMap.erase(p);
        return value;
    }

    virtual bool remove(const Key_T& key)
    {
        Iter_T p = mMap.find(key);
        if (p == mMap.end())
            return false;
        mDestructor(p->second);
        mMap.erase(p);
        return true;
    }

    virtual Value_T& operator[](const Key_T& key)
    {
        return mMap[key];
    }

    virtual void clear()
    {
        destroy();
    }

    virtual Map_T* clone() const
    {
        TrueFilter<Pair> identity;
        return filter(identity);
    }

    virtual Map_T* filter(Filter<Pair>& filter) const
    {
        Map_T* filtered = new Map_T;
        Iterator it = iterator();
        while (it->hasNext())
        {
            Pair p = it->next();
            if (filter(p))
            {
                filtered->put(p.first, mCloner(p.second));
            }
        }
        return filtered;
    }

protected:
    typedef std::map<Key_T, Value_T> Storage_T;
    typedef typename Storage_T::iterator Iter_T;
    typedef typename Storage_T::const_iterator ConstIter_T;
    Storage_T mMap;
    Cloner_T mCloner;
    Destructor_T mDestructor;

    void destroy()
    {
        Iter_T it = mMap.begin(), end = mMap.end();
        for(;it != end; ++it)
        {
            mDestructor(it->second);
        }
        mMap.clear();
    }
};

}

#endif
