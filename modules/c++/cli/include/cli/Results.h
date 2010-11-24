/* =========================================================================
 * This file is part of cli-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2010, General Dynamics - Advanced Information Systems
 *
 * cli-c++ is free software; you can redistribute it and/or modify
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

#ifndef __CLI_RESULTS_H__
#define __CLI_RESULTS_H__

#include <map>
#include "cli/Value.h"

namespace cli
{

class Results
{
public:
    Results()
    {
    }
    ~Results()
    {
        destroy();
    }

    bool exists(const std::string& key) const
    {
        return mMap.find(key) != mMap.end();
    }

    size_t size() const
    {
        return mMap.size();
    }

    cli::Value* operator[](const std::string& key) const
            throw (except::NoSuchKeyException)
    {
        return getValue(key);
    }

    cli::Value* getValue(const std::string& key) const
            throw (except::NoSuchKeyException)
    {
        ConstIter_T p = mMap.find(key);
        if (p == mMap.end())
            throw except::NoSuchKeyException(Ctxt(key));
        return p->second;
    }

    template <typename T>
    T get(const std::string& key, unsigned int index = 0) const
            throw (except::NoSuchKeyException)
    {
        return getValue(key)->get<T>(index);
    }

    template <typename T>
    T operator()(const std::string& key, unsigned int index = 0) const
            throw (except::NoSuchKeyException)
    {
        return get<T>(key, index);
    }

    std::vector<std::string> keys() const
    {
        std::vector<std::string> vec(mMap.size());
        ConstIter_T p = mMap.begin();
        for(size_t i = 0; p != mMap.end(); ++p, ++i)
            vec[i] = p->first;
        return vec;
    }

protected:
    typedef std::map<std::string, cli::Value*> Storage_T;
    typedef Storage_T::iterator Iter_T;
    typedef Storage_T::const_iterator ConstIter_T;
    Storage_T mMap;

    void destroy()
    {
        Iter_T it = mMap.begin(), end = mMap.end();
        for (; it != end; ++it)
        {
            delete it->second;
        }
        mMap.clear();
    }

    friend class ArgumentParser;

    void put(const std::string& key, cli::Value *value)
    {
        if (exists(key))
            delete getValue(key);
        mMap[key] = value;
    }
};

}

#endif
