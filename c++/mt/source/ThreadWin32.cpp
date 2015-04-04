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
#include "mt/ThreadWin32.h"


mt::ThreadWin32::~ThreadWin32()
{
    if (mNative)
    {
        CloseHandle(mNative);
    }

}

void mt::ThreadWin32::start()
{
    DWORD threadId;

    mNative = __CREATETHREAD(NULL,
                             0,
                             __start,
                             (void*)this,
                             0,
                             &threadId);
    if (mNative == NULL)
        throw mt::ThreadResourceException("Thread creation failed");


}


void mt::ThreadWin32::join()
{
    if (WaitForSingleObject(mNative, INFINITE) == WAIT_FAILED)
        throw mt::ThreadResourceException("Thread join failed");

}

#endif // Not using another package

#endif // We're on windows
