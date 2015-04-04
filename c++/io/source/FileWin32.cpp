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

#ifdef WIN32

#include <limits>
#include <cmath>
#include "io/File.h"

void io::File::create(const std::string& str,
                       int accessFlags,
                       int creationFlags)
{
    // If the truncate bit is on AND the file does exist,
    // we need to set the mode to TRUNCATE_EXISTING
    if ((creationFlags & io::File::TRUNCATE) && sys::OS().exists(str) )
    {
        creationFlags = TRUNCATE_EXISTING;
    }
    else
    {
        creationFlags = ~io::File::TRUNCATE & creationFlags;
    }

    mHandle = CreateFile(str.c_str(),
                         accessFlags,
                         FILE_SHARE_READ, NULL,
                         creationFlags,
                         FILE_ATTRIBUTE_NORMAL, NULL);

    if (mHandle == SYS_INVALID_HANDLE)
    {
        throw sys::SystemException(Ctxt(FmtX("Error opening file: [%s]", str.c_str())));
    }
    mPath = str;
}

void io::File::readInto(char *buffer, sys::Size_T size)
{
    static const DWORD MAX_READ_SIZE = std::numeric_limits<DWORD>::max();
    size_t bytesRead = 0;
    size_t bytesRemaining = size;

    while (bytesRead < size)
    {
        // Determine how many bytes to read
        const DWORD bytesToRead =
            std::min<DWORD>(MAX_READ_SIZE, bytesRemaining);

        // Read from file
        DWORD bytesThisRead = 0;
        if (!ReadFile(mHandle,
                      buffer + bytesRead,
                      bytesToRead,
                      &bytesThisRead,
                      NULL))
        {
            throw sys::SystemException(Ctxt("Error reading from file"));
        }
        else if (bytesThisRead == 0)
        {
            //! ReadFile does not fail when finding the EOF --
            //  instead it reports 0 bytes read, so this stops an infinite loop
            //  from Unexpected EOF
            throw sys::SystemException(Ctxt("Unexpected end of file"));
        }

        bytesRead += bytesThisRead;
        bytesRemaining -= bytesThisRead;
    }
}

void io::File::writeFrom(const char *buffer, sys::Size_T size)
{
    static const DWORD MAX_WRITE_SIZE = std::numeric_limits<DWORD>::max();
    size_t bytesRemaining = size;
    size_t bytesWritten = 0;

    while (bytesWritten < size)
    {
        // Determine how many bytes to write
        const DWORD bytesToWrite =
            std::min<DWORD>(MAX_WRITE_SIZE, bytesRemaining);

        // Write the data
        DWORD bytesThisWrite = 0;
        if (!WriteFile(mHandle,
                       buffer + bytesWritten,
                       bytesToWrite,
                       &bytesThisWrite,
                       NULL))
        {
            throw sys::SystemException(Ctxt("Writing from file"));
        }

        // Accumulate this write until we are done
        bytesRemaining -= bytesThisWrite;
        bytesWritten += bytesThisWrite;
    }
}

sys::Off_T io::File::seekTo(sys::Off_T offset, int whence)
{
    /* Ahhh!!! */
    LARGE_INTEGER largeInt;
    LARGE_INTEGER toWhere;
    largeInt.QuadPart = offset;
    if (!SetFilePointerEx(mHandle, largeInt, &toWhere, whence))
        throw sys::SystemException(Ctxt("SetFilePointer failed"));

    return (sys::Off_T) toWhere.QuadPart;
}

sys::Off_T io::File::length()
{
    sys::OS os;
    return os.getSize(mPath);
}

sys::Off_T io::File::lastModifiedTime()
{
    sys::OS os;
    return os.getLastModifiedTime(mPath);
}

void io::File::flush()
{
    if (!FlushFileBuffers(mHandle))
    {
        throw sys::SystemException(Ctxt("Error flushing file " + mPath));
    }
}

void io::File::close()
{
    CloseHandle(mHandle);
    mHandle = SYS_INVALID_HANDLE;
}

#endif
