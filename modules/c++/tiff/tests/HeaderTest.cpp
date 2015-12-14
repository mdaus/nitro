/* =========================================================================
 * This file is part of tiff-c++ 
 * =========================================================================
 * 
 * (C) Copyright 2004 - 2014, MDA Information Systems LLC
 *
 * tiff-c++ is free software; you can redistribute it and/or modify
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

#include <import/except.h>
#include <import/io.h>
#include <import/tiff.h>
#include <fstream>

using namespace sys;

int main()
{
    try
    {
        io::StandardOutStream st;
        tiff::IFDEntry entry(254, 1);
        entry.print(st);
        entry.print(st);

        tiff::IFDEntry entry2(255, 1);
        entry2.print(st);
        entry2.print(st);
    }
    catch (except::Throwable& t)
    {
        std::cerr << "Caught throwable: " << t.toString() << std::endl;
        exit( EXIT_FAILURE);
    }
    catch (...)
    {
        std::cerr << "Caught unnamed exception" << std::endl;
    }
    return 0;
}
