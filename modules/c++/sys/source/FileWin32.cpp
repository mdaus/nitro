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


#ifdef WIN32

#include "sys/File.h"

void sys::File::create(const std::string& str,
                       int accessFlags,
                       int creationFlags)
{
    // The user should pass in the info telling it
    // to truncate
    //if (accessFlags & sys::File::WRITE_ONLY)
    //{
    //WIN32_FIND_DATA findData;
    //mHandle = FindFirstFile(str.c_str(), &findData);
    //if (mHandle != SYS_INVALID_HANDLE)
    //{
    //creationFlags |= TRUNCATE_EXISTING;
    //}
    //}

    // If the truncate bit is on AND the file does exist,
    // we need to set the mode to TRUNCATE_EXISTING
    if ((creationFlags & sys::File::TRUNCATE) && sys::OS().exists(str) )
    {
	creationFlags = TRUNCATE_EXISTING;
    }
    else
    {
	creationFlags = ~sys::File::TRUNCATE & creationFlags;
    }

    mHandle = CreateFile(str.c_str(),
                         accessFlags,
                         FILE_SHARE_READ, NULL,
                         creationFlags,
                         FILE_ATTRIBUTE_NORMAL, NULL);

    if (mHandle == SYS_INVALID_HANDLE)
    {
        throw sys::SystemException("Error opening file: " + str);
    }
    mFileName = str;
}

void sys::File::readInto(char *buffer, Size_T size)
{
    /****************************
     *** Variable Declarations ***
     ****************************/
    DWORD bytesRead = 0;        /* Number of bytes read during last read operation */
    DWORD totalBytesRead = 0;   /* Total bytes read thus far */

    /* Make the next read */
    if (!ReadFile(mHandle, buffer, size, &bytesRead, 0))
    {
        throw sys::SystemException("Error reading from file");
    }
}



void sys::File::writeFrom(const char *buffer, Size_T size)
{
    DWORD actuallyWritten = 0;

    do
    {
        /* Keep track of the bytes we read */
        DWORD bytesWritten;
        /* Write the data */
        BOOL ok = WriteFile(mHandle, buffer, size, &bytesWritten, NULL);
        if (!ok)
        {
            /* If the function failed, we want to get the last error */
            throw sys::SystemException("Writing from file");
        }
        /* Otherwise, we want to accumulate this write until we are done */
        actuallyWritten += bytesWritten;
    }
    while (actuallyWritten < size);
}

sys::Off_T sys::File::seekTo(sys::Off_T offset, int whence)
{
    /* Ahhh!!! */
    LARGE_INTEGER largeInt;
    int lastError;
    LONG low = offset;
    PLONG high = NULL;
    if (offset > 0)
    {
        largeInt.QuadPart = offset;
        low = largeInt.LowPart;
        high = &largeInt.HighPart;
    }
    largeInt.LowPart = SetFilePointer(mHandle, low, high, whence);

    lastError = GetLastError();
    if ((largeInt.LowPart == INVALID_SET_FILE_POINTER) &&
            (lastError != NO_ERROR))
    {
        /*LPVOID lpMsgBuf;
        LPVOID lpDisplayBuf;

        FormatMessage(
         FORMAT_MESSAGE_ALLOCATE_BUFFER | 
         FORMAT_MESSAGE_FROM_SYSTEM |
         FORMAT_MESSAGE_IGNORE_INSERTS,
         NULL,
         lastError,
         MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
         (LPTSTR) &lpMsgBuf,
         0, NULL );

        MessageBox(NULL, (LPCTSTR)lpMsgBuf, TEXT("Error"), MB_OK); 

        LocalFree(lpMsgBuf);
        LocalFree(lpDisplayBuf);*/

        throw sys::SystemException("SetFilePointer failed");
    }
    return (sys::Off_T) largeInt.QuadPart;
}


sys::Off_T sys::File::length()
{
    return sys::OS().getSize(mFileName);
}

void sys::File::close()
{
    CloseHandle(mHandle);
    mHandle = SYS_INVALID_HANDLE;
}
#endif
