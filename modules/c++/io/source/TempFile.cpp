/* =========================================================================
 * This file is part of io-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2016, MDA Information Systems LLC
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

#include <cstdio>
#include <string>
#include <fstream>
#include <import/except.h>
#include "sys/Conf.h"
#include "io/TempFile.h"

io::TempFile::TempFile(bool destroy, const std::string& path) :
    mDestroy(destroy),
    mOS(sys::OS()),
    mName(mOS.getTempName(path))
{
    if (mName.empty())
    {
        throw except::Exception(Ctxt("Unable to create temporary file"));
    }
}

io::TempFile::~TempFile()
{
    try
    {
        if (mDestroy && mOS.exists(mName))
        {
            mOS.remove(mName);
        }
    }
    catch (...)
    {
        // Do nothing
    }
}

std::string io::TempFile::pathname() const
{
    return mName;
}

