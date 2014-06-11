/* =========================================================================
 * This file is part of io-c++ 
 * =========================================================================
 * 
 * (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
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

#include <io/SafePath.h>
#include <unique/UUID.hpp>
#include <sys/OS.h>

namespace io
{

SafePath::SafePath(const std::string& realPathname) :
    mRealPathname(realPathname),
    mTempPathname(unique::generateUUID() + ".tmp"),
    moved(false)
{
}

SafePath::~SafePath()
{
    try
    {
        sys::OS os;
        moveFile();
        if(!moved)
        {
            os.remove(mTempPathname);
        }
    }
    catch(...){}
}

std::string SafePath::getTempPathname() const
{
    if(moved)
    {
        throw except::Exception(Ctxt(
                "File has already been moved, use of getTemp() is invalid."));
    }
    return mTempPathname;
}

void SafePath::moveFile()
{
    if(!moved)
    {
        sys::OS os;
        if(os.move(mTempPathname, mRealPathname))
        {
            moved = true;
        }
        else
        {
            throw except::Exception(Ctxt(
                           "Error renaming file from " + mTempPathname + " to " 
                           + mRealPathname + " in moveFile()."));
        }
    }
}

}
