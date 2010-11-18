/* =========================================================================
 * This file is part of xstr-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
 *
 * xstr-c++ is free software; you can redistribute it and/or modify
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
#include <import/xstr.h>
#include <import/sys.h>

using namespace xstr;

int main(int argc, char **argv)
{
    try
    {
        String s;
        s = "pirates";
        std::cout << s << std::endl;

        //should print out rat
        std::cout << s.substring(2, 5) << std::endl;

        std::cout << s.startsWith("pi") << std::endl;
        std::cout << s.endsWith("nope") << std::endl;

        s = String::valueOf(1);
        std::cout << s << std::endl;

        s = 1.21f;
        std::cout << s << std::endl;

        for (String::iterator it = s.begin(); it != s.end(); ++it)
            std::cout << *it << std::endl;

        s = "All work and no play makes Tom a dull\nboy";
        std::cout << s.toUpperCase() << std::endl;
        std::vector < String > parts = s.split("\\s");
        std::cout << parts.size() << std::endl;
        for (std::vector<String>::iterator it = parts.begin(); it
                != parts.end(); ++it)
            std::cout << *it << std::endl;

        std::cout << s.matches(".*Tom.*") << std::endl;
        std::cout << s.matches(".*Goof.*") << std::endl;

        std::cout << s.contains('z') << std::endl;
        std::cout << (s.indexOf('z') == String::npos) << std::endl;
        std::cout << s.indexOf("dull") << std::endl;
        std::cout << String("real american hero").lastIndexOf(String('r'))
                << std::endl;

        s = "gonnaBlowUp";
        std::cout << s.substring(2, 1) << std::endl;
    }
    catch (except::Exception& e)
    {
        std::cout << "Error: " << e.toString() << std::endl;
    }
}
