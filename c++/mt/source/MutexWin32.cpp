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

#include "mt/MutexWin32.h"

mt::MutexWin32::MutexWin32()
{
    mNative = CreateMutex(NULL, FALSE, NULL);
    if (mNative == NULL)
        throw mt::ThreadResourceException("Mutex initializer failed");
}

mt::MutexWin32::~MutexWin32()
{
    CloseHandle(mNative);
}

void mt::MutexWin32::lock()
{
    if (WaitForSingleObject(mNative, INFINITE) == WAIT_FAILED)
        throw mt::LockException("Mutex lock failed");
}

void mt::MutexWin32::unlock()
{
    if (ReleaseMutex(mNative) != TRUE)
        throw mt::LockException("Mutex unlock failed");
}

HANDLE& mt::MutexWin32::getNative()
{
    return mNative;
}

#endif // Not some other thread package

#endif // Windows platform
