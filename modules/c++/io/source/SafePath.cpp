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
#include <sys/Path.h>

namespace io
{

SafePath::SafePath(const std::string& realPathname) :
    mRealPathname(realPathname),
    mTempPathname(sys::Path::joinPaths(
                  sys::Path::splitPath(realPathname).first, 
                  unique::generateUUID() + ".tmp")),
    mMoved(false)
{
}

SafePath::~SafePath()
{
    try
    {
        try
        {
            moveFile();
        }
        catch(...)
        {
            sys::OS os;
            if(os.exists(mTempPathname))
            {
                os.remove(mTempPathname);
            }
        }
    }
    catch(...)
    {
        /* do nothing */
    }
}

std::string SafePath::getTempPathname() const
{
    if(mMoved)
    {
        throw except::Exception(Ctxt(
            "File has already been moved, \
             use of getTempPathname() is invalid."));
    }
    return mTempPathname;
}

void SafePath::moveFile()
{
    if(!mMoved)
    {
        sys::OS os;
        if(os.move(mTempPathname, mRealPathname))
        {
            mMoved = true;
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
