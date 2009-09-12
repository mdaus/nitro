/* =========================================================================
 * This file is part of re-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
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

#include <iostream>
#include <vector>
#include <string>
#include <import/except.h>
#include <import/str.h>
#include <import/re.h>

using namespace except;
using namespace str;
using namespace re;

using namespace std;

int main()
{
    try
    {
        PCRE rx;
        rx.compile("^([^:]+):[ ]*([^\r\n]+)\r\n(.*)");

        std::cout << "1) Performing HTTP header test..." << std::endl;

        PCREMatch matches;
        if (rx.match("Proxy-Connection: Keep-Alive\r\n", matches))
        {
            cout << "    Found " << matches.size() << " regex matches" << endl;
            if (matches.size() != 4)
                throw PCREException("Could not match HTTP header kv pair");
        }
        else
        {
            throw PCREException("Could not match regex");
        }

        std::cout << "2) Performing sub-string match test..." << std::endl;

        PCRE rx2;
        rx2.compile("ar");
        PCREMatch subs2;
        rx2.searchAll("arabsdsarbjudarc34ardnjfsdveqvare3arfarg", subs2);

        if ( subs2.size() == 7 )
            std::cout << "    Found all sub-strings" << std::endl;
        else
            throw PCREException("Did not find a sub-string");

        cout << "3) Peforming 'sub' test..." << endl;

        std::string subst = rx2.sub("Hearo", "ll");

        if (subst == "Hello")
            std::cout << "    Successfully substituted string pattern" << std::endl;
        else
            throw PCREException("Did not correctly substitute string pattern");

        subst = rx2.sub("Hearo Keary!", "ll");
        if (subst == "Hello Kelly!")
            std::cout << "    Successfully substituted string pattern" << std::endl;
        else
        {
            std::cout << subst << std::endl;
            throw PCREException("Did not correctly substitute string pattern");
        }

        cout << "4) Performing 'split' test..." << endl;
        cout << "    Delimiter is 'ar'" << std::endl;

        std::vector< std::string > vec;
        rx2.split("ONEarTWOarTHREE", vec);
        std::vector< std::string >::iterator iter;
        int i = 0;
        for (iter = vec.begin(); iter != vec.end(); ++iter)
        {
            std::cout << "    " << *iter << std::endl;
            if (i == 0)
            {
                if (*iter != "ONE")
                    throw PCREException("First match was supposed to be ONE");
                i++;
            }
            else if (i == 1)
            {
                if (*iter != "TWO")
                    throw PCREException("Second match was supposed to be TWO");
                i++;
            }
            else if (i == 2)
            {
                if (*iter != "THREE")
                    throw PCREException("Third match was supposed to be THREE");
                i++;
            }
            else
            {
                throw PCREException("Too many matches!");
            }

        }

        std::cout << "Test Passed!" << std::endl;

        return 0;
    }
    catch (except::Throwable& t)
    {
        //std::cout << t.getMessage() << std::endl;
        std::cout << "ERROR!!!!" << t.getTrace() << std::endl;
    }
}
