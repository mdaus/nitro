/* =========================================================================
 * This file is part of sys-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
 *
 * sys-c++ is free software; you can redistribute it and/or modify
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

#include <iostream>

#include <sys/ScopedCopyablePointer.h>
#include <except/Exception.h>

namespace
{
struct Foo
{
    Foo()
    : val1(0),
      val2(0)
    {
    }

    size_t val1;
    size_t val2;
};

struct Bar
{
    Bar()
    : val3(0)
    {
    }

    sys::ScopedCopyablePointer<Foo> foo;
    size_t                          val3;
};
}

int main(int argc, char** argv)
{
    try
    {
        // Initialize the values
        Bar bar1;
        bar1.foo.reset(new Foo());
        bar1.foo->val1 = 10;
        bar1.foo->val2 = 20;
        bar1.val3 = 30;

        // Show that the compiler-generated copy constructor is correct
        Bar bar2(bar1);
        if (!(bar2.foo->val1 == 10 &&
              bar2.foo->val2 == 20 &&
              bar2.val3 == 30))
        {
            std::cerr << "Copy constructor is invalid!\n";
            return 1;
        }

        // Show it was a deep copy
        bar2.foo->val1 = 40;
        bar2.foo->val2 = 50;
        bar2.val3 = 60;
        if (!(bar1.foo->val1 == 10 &&
              bar1.foo->val2 == 20 &&
              bar1.val3 == 30))
        {
            std::cerr << "Copy constructor is not a deep copy!\n";
            return 1;
        }

        // Show that the assignment operator is correct
        bar2 = bar1;
        if (!(bar2.foo->val1 == 10 &&
              bar2.foo->val2 == 20 &&
              bar2.val3 == 30))
        {
            std::cerr << "Assignment operator is invalid!\n";
            return 1;
        }

        // Show it was a deep copy
        bar2.foo->val1 = 40;
        bar2.foo->val2 = 50;
        bar2.val3 = 60;
        if (!(bar1.foo->val1 == 10 &&
              bar1.foo->val2 == 20 &&
              bar1.val3 == 30))
        {
            std::cerr << "Assignment operator is not a deep copy!\n";
            return 1;
        }

        // TODO Show that ScopedCopyablePointer deletes its memory.
        //      For now, can run valgrind to show this.

        std::cout << "All tests passed\n";
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Caught std::exception: " << ex.what() << std::endl;
        return 1;
    }
    catch (const except::Exception& ex)
    {
        std::cerr << "Caught except::exception: " << ex.getMessage()
                  << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "Caught unknown exception\n";
        return 1;
    }

    return 0;
}
