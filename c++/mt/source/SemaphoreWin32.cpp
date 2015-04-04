/* =========================================================================
 * This file is part of mt-c++ 
 * =========================================================================
 * 
 * (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
 *
 * mt-c++ is free software; you can redistribute it and/or modify
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


#if defined(WIN32)

#if !defined(USE_NSPR_THREADS)

#include "mt/SemaphoreWin32.h"

mt::SemaphoreWin32::SemaphoreWin32(unsigned int count)
{
    mNative = CreateSemaphore(NULL, count, MAX_COUNT, NULL);
    if (mNative == NULL)
        throw mt::ThreadResourceException("CreateSempaphore Failed");

}

void mt::SemaphoreWin32::wait()
{
    DWORD waitResult = WaitForSingleObject(
                           mNative,
                           INFINITE);
    if (waitResult != WAIT_OBJECT_0)
    {
        throw mt::ThreadResourceException("Semaphore wait failed");
    }
}

void mt::SemaphoreWin32::signal()
{
    if (!ReleaseSemaphore(mNative,
                          1,
                          NULL) )
    {
        throw mt::ThreadResourceException("Semaphore signal failed");
    }
}

HANDLE& mt::SemaphoreWin32::getNative()
{
    return mNative;
}

#endif
#endif
