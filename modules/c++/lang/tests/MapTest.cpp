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

#include <import/lang.h>
#include <import/sys.h>
using namespace lang;

typedef Pair<std::string, std::string>MapPair;
typedef STLMap<std::string, std::string>MutableDictionary;
typedef const MutableDictionary ImmutableDictionary;

typedef HashMap<std::string, std::string, DefaultStringHash<std::string>,
        DefaultComparator<std::string> > MutableHashMap;

class MyFilter : public Filter<MutableDictionary::Pair>
{
public:
    MyFilter()
    {
    }
    virtual ~MyFilter()
    {
    }
    bool operator()(const MutableDictionary::Pair& val)
    {
        return str::startsWith(val.first, "my-");
    }
};

template<typename Iterator_T>
void printDict(Iterator_T it)
{
    while (it->hasNext())
    {
        MapPair val = it->next();
        std::cout << val.first << " = " << val.second << std::endl;
    }
}

template<typename Dict_T>
void testDict()
{
    Dict_T dict;
    dict.put("my-name", "tom");
    dict.put("my-number", "15");
    dict.put("another", "string value");
    std::cout << "pre-filtered size: " << dict.size() << std::endl;
    MyFilter scalarFilter;
    Dict_T *filtered = dict.filter(scalarFilter);
    std::cout << "filtered size: " << filtered->size() << std::endl;
    printDict(filtered->iterator());
    filtered->clear();
    std::cout << "filtered size after clear: " << filtered->size() << std::endl;
    delete filtered;

    filtered = dict.clone();
    std::cout << "filtered size after clone: " << filtered->size() << std::endl;

    std::string v = filtered->pop("my-name");
    std::cout << "Popped: " << v << std::endl;
    std::cout << "filtered size after pop: " << filtered->size() << std::endl;
    delete filtered;
}

int main()
{
    try
    {
        testDict<MutableDictionary>();
        testDict<MutableHashMap>();
    }
    catch (except::Exception& e)
    {
        std::cout << "Error: " << e.getMessage() << std::endl;
        return 1;
    }
    return 0;
}
