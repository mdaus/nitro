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


#include <import/sys.h>
#include <fstream>
#include <iomanip>

using namespace sys;

int main(int argc, char **argv)
{
    try
    {
        sys::OS os;
        std::string fileName = os.getCurrentWorkingDirectory() + 
                                 os.getDelimiter() + ".." +
                                 os.getDelimiter() + "blah.txt";

        Path::StringPair parts = Path::splitPath(fileName);
        std::cout << parts.first << " --- " << parts.second << std::endl;
        
        parts = Path::splitDrive("c:/junk.txt");
        std::cout << parts.first << " --- " << parts.second << std::endl;
        
        std::string base = 
            Path::basename("/data/nitf///data/vendor1.ntf", true);
        std::cout << base << std::endl;


        parts = Path::splitPath("/data.txt");
        std::cout << parts.first << " --- " << parts.second << std::endl;
        
        parts = Path::splitExt(fileName);
        std::cout << parts.first << " --- " << parts.second << std::endl;
        
        parts = Path::splitExt(Path::splitPath(fileName).second);
        std::cout << parts.first << " --- " << parts.second << std::endl;
        
        std::cout << Path::normalizePath(fileName) << std::endl;
        std::cout << Path::normalizePath("c:/data/nitf/data/vendor1.ntf") << std::endl;
        std::cout << Path::normalizePath("/data/nitf///data/vendor1.ntf") << std::endl;
        
        

        std::cout << Path::normalizePath("/data/nitf///data/../vendor1.ntf") << std::endl;
        
        std::cout << Path::normalizePath("../data/../../..//./nitf///data/../vendor1.ntf") << std::endl;
        
        std::cout << Path::normalizePath("data/junk/tzellman/../../../../../..///./nitf///data/../vendor1.ntf") << std::endl;
        
        std::cout << Path::joinPaths("/data/junk", "test.txt") << std::endl;
        std::cout << Path::absolutePath("data/junk/tzellman/../../../../../..///./nitf///data/../vendor1.ntf") << std::endl;
        std::cout << Path::normalizePath("data/junk/tzellman/../../../../../..///./nitf///data/../vendor1.ntf") << std::endl;
        std::cout << Path::absolutePath("c:/data/junk/tzellman/../../../../../..///./nitf///data/../vendor1.ntf") << std::endl;
        std::cout << Path::normalizePath("c:/../../../junk.txt") << std::endl;
        
        std::cout << Path::absolutePath("/home/tzellman/dev/") << std::endl;
        

    }
    catch (except::Throwable& t)
    {
        std::cerr << "Caught throwable: " << t.getMessage() << " (type:" << t.getType() << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
    catch (...)
    {
        std::cerr << "Caught unnamed exception" << std::endl;
    }
    return 0;
}
