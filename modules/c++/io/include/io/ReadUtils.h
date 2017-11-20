/* =========================================================================
 * This file is part of io-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2017, MDA Information Systems LLC
 *
 * io-c++ is free software; you can redistribute it and/or modify
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

#ifndef __IO_READ_UTILS_H__
#define __IO_READ_UTILS_H__

#include <string>
#include <vector>

#include <sys/Conf.h>

namespace io
{
void readFileContents(const std::string& pathname,
                      std::vector<sys::byte>& buffer);

void readFileContents(const std::string& pathname, std::string& str);

inline
std::string readFileContents(const std::string& pathname)
{
    std::string str;
    readFileContents(pathname, str);
    return str;
}
}

#endif
