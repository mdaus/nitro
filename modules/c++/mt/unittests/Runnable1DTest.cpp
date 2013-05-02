/* =========================================================================
 * This file is part of mt-c++ 
 * =========================================================================
 * 
 * (C) Copyright 2004 - 2013, General Dynamics - Advanced Information Systems
 *
 * mt-c++ is free software; you can redistribute it and/or modify
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

#include "import/sys.h"
#include "import/mt.h"
#include "TestCase.h"

using namespace sys;
using namespace mt;
using namespace std;

class AddOp
{
public:
    AddOp()
    {
    }

    void operator()(size_t element) const
    {
        std::cout << element << std::endl;
    }
};

TEST_CASE(Runnable1DTest)
{
    std::cout << "Running test case" << std::endl;
    const AddOp op;
    std::cout << "Calling run1D" << std::endl;
    run1D(10, 16, op);
}

TEST_CASE(Runnable1DWithCopiesTest)
{
    // TODO: Need an actual test case that shows the threads all truly have
    //       their own local storage which isn't colliding
    std::cout << "Running test case" << std::endl;
    const AddOp op;
    std::cout << "Calling run1D" << std::endl;
    run1DWithCopies(10, 16, op);
}

int main(int argc, char *argv[])
{
    TEST_CHECK(Runnable1DTest);
    TEST_CHECK(Runnable1DWithCopiesTest);

    return 0;
}
