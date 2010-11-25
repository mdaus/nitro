/* =========================================================================
 * This file is part of str-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
 *
 * str-c++ is free software; you can redistribute it and/or modify
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

#include <import/str.h>
#include "TestCase.h"

TEST_CASE(testNull)
{
    std::string s = str::toString(NULL);
}

TEST_CASE(testTrim)
{
    std::string s = "  test   ";
    str::trim( s);
    TEST_ASSERT_EQ(s, "test");
}

TEST_CASE(testUpper)
{
    std::string s = "test-something1";
    str::upper( s);
    TEST_ASSERT_EQ(s, "TEST-SOMETHING1");
}

TEST_CASE(testLower)
{
    std::string s = "TEST1";
    str::lower( s);
    TEST_ASSERT_EQ(s, "test1");
}

TEST_CASE(testSplit)
{
    std::string s = "space delimited values are the best!";
    std::vector<std::string> parts = str::split(s, " ");
    TEST_ASSERT_EQ(parts.size(), 6);
    parts = str::split(s, " ", 3);
    TEST_ASSERT_EQ(parts.size(), 3);
    TEST_ASSERT_EQ(parts[2], "values are the best!");
}

int main(int argc, char* argv[])
{
    TEST_CHECK( testNull);
    TEST_CHECK( testTrim);
    TEST_CHECK( testUpper);
    TEST_CHECK( testLower);
    TEST_CHECK( testSplit);
}
