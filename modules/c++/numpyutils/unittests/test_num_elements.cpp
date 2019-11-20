/* =========================================================================
 * This file is part of numpyutils-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2019, MDA Information Systems LLC
 *
 * math-c++ is free software; you can redistribute it and/or modify
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

#include <TestCase.h>
#include <numpyutils/numpyutils.h>

namespace
{
TEST_CASE(testGetNumElements)
{
    std::vector<int> data(50);
    PyObject* pyArrayObject = numpyutils::toNumpyArray(5, 10, NPY_INT, data.data());

    TEST_ASSERT_EQ(50, numpyutils::getNumElements(pyArrayObject));

    PyObject* nonArrayObject = PyString_FromString("not an array");
    TEST_THROWS(numpyutils::getNumElements(nonArrayObject));
}
}

int main()
{
    TEST_CHECK(testGetNumElements);
    return 0;
}

