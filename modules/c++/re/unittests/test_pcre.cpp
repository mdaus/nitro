/* =========================================================================
 * This file is part of re-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2014, MDA Information Systems LLC
 *
 * re-c++ is free software; you can redistribute it and/or modify
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

#include <import/re.h>
#include "TestCase.h"

TEST_CASE(testMatches)
{
    re::PCREMatch matches;
    re::PCRE rx("^([^:]+):[ ]*([^\r\n]+)\r\n(.*)");
    TEST_ASSERT(rx.match("Proxy-Connection: Keep-Alive\r\n", matches));
    TEST_ASSERT_EQ(matches.size(), 4);
}

TEST_CASE(testSearch)
{
    re::PCREMatch matches;
    re::PCRE rx("ar");
    rx.searchAll("arabsdsarbjudarc34ardnjfsdveqvare3arfarg", matches);
    TEST_ASSERT_EQ(matches.size(), 7);
}

TEST_CASE(testDotAllFlag)
{
    // This should match both the "3.3" and "4\n2"
    re::PCREMatch matches1;
    re::PCRE rx1("\\d.\\d", re::PCRE::PCRE_DOTALL);
    rx1.searchAll("3.3 4\n2", matches1);
    TEST_ASSERT_EQ(matches1.size(), 2);

    // This should only match the "3.3"
    re::PCREMatch matches2;
    re::PCRE rx2("\\d.\\d", re::PCRE::PCRE_NONE);
    rx2.searchAll("3.3 4\n2", matches2);
    TEST_ASSERT_EQ(matches2.size(), 1);

    // This should only match the "3.3" if the replace_dot() function
    // is working correctly
    re::PCREMatch matches3;
    re::PCRE rx3("\\d\\.\\d", re::PCRE::PCRE_DOTALL);
    rx3.searchAll("3.3 4\n2", matches3);
    TEST_ASSERT_EQ(matches3.size(), 1);
}

TEST_CASE(testSub)
{
    re::PCREMatch matches;
    re::PCRE rx("ar");
    std::string subst = rx.sub("Hearo", "ll");
    TEST_ASSERT_EQ(subst, "Hello");
    subst = rx.sub("Hearo Keary!", "ll");
    TEST_ASSERT_EQ(subst, "Hello Kelly!");
}

TEST_CASE(testSplit)
{
    re::PCREMatch matches;
    re::PCRE rx("ar");
    std::vector<std::string> vec;
    rx.split("ONEarTWOarTHREE", vec);
    TEST_ASSERT_EQ(vec.size(), 3);
    TEST_ASSERT_EQ(vec[0], "ONE");
    TEST_ASSERT_EQ(vec[1], "TWO");
    TEST_ASSERT_EQ(vec[2], "THREE");
}

int main(int, char**)
{
    TEST_CHECK( testMatches);
    TEST_CHECK( testSearch);
    TEST_CHECK( testDotAllFlag);
    TEST_CHECK( testSub);
    TEST_CHECK( testSplit);
}
