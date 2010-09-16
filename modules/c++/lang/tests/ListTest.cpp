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

typedef STLList<std::string> StringList;
typedef STLVector<std::string> StringVector;

template <typename Iterator_T>
void printList(Iterator_T it)
{
    while (it->hasNext())
    {
        std::string v = it->next();
        std::cout << v << std::endl;
    }
}

template<typename List_T>
void testStringList()
{
    std::cout << "===Testing List===" << std::endl;
    List_T list;
    list.add("tom");
    list.add("dan");
    list.add("adam");
    list.add("chris");
    printList(list.iterator());
    //printList(list.iterator(2, 1));
    //does not support negative indexing yet
    std::cout << "size: " << list.size() << std::endl;

    //show off using the interface
    Collection<std::string>* coll = new List_T;
    coll->add("something");
    delete coll;

    std::cout << "Cloning..." << std::endl;
    List_T *newList = list.clone();
    printList(newList->iterator());
    newList->clear();
    delete newList;

    std::cout << "Removing..." << std::endl;
    list.remove("chris");
    printList(list.iterator());
    list.remove("dan");
    printList(list.iterator());
    list.remove("tom");
    printList(list.iterator());

    std::cout << "Inserting..." << std::endl;
    list.insert("dan", 0);
    list.insert("tom", 1);
    list.insert("chris", 0);
    printList(list.iterator());

    std::cout << "Inserting2..." << std::endl;
    List_T list2;
    list2.insert("dan", 0);
    list2.insert("tom", 1);
    list2.insert("chris", 0);
    printList(list2.iterator());
    list2.clear();

    for(size_t i = 0; i < 100; ++i)
    {
        list2.insert(FmtX("%d", i));
    }
    std::cout << list2.size() << std::endl;
}

int main()
{
    try
    {
        testStringList<StringList> ();
        testStringList<StringVector> ();
    }
    catch (except::Exception& e)
    {
        std::cout << "Error: " << e.getMessage() << std::endl;
        return 1;
    }
    return 0;
}
