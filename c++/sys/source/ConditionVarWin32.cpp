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


#if defined(WIN32) && defined(_REENTRANT)

#if !defined(USE_NSPR_THREADS) && !defined(__POSIX)

#include "sys/ConditionVarWin32.h"

sys::ConditionVarDataWin32::ConditionVarDataWin32(): nWaiters(0), nRelease(0), nWaitGeneration(0)
{
    InitializeCriticalSection(&lWaiters);
    event = CreateEvent(NULL, TRUE, FALSE, NULL);
    if (event == NULL)
        throw sys::SystemException("ConditionVarDataWin32 Initializer failed");
}

sys::ConditionVarDataWin32::~ConditionVarDataWin32()
{
    CloseHandle(event);
    DeleteCriticalSection(&lWaiters);
}

bool sys::ConditionVarWin32::wait(double timeout)
{
    dbg_printf("Timed waiting on condition [%f]\n", timeout);
    if (timeout != 0)
    {
        EnterCriticalSection(&mNative.lWaiters);

        mNative.nWaiters++;

        int myGeneration = mNative.nWaitGeneration;

        LeaveCriticalSection(&mNative.lWaiters);
        mMutex->unlock();

        for (;;)
        {
            WaitForSingleObject(mNative.event, (int)timeout * 1000);
            EnterCriticalSection(&mNative.lWaiters);

            bool waitDone = mNative.nRelease > 0 &&
                            mNative.nWaitGeneration != myGeneration;
            LeaveCriticalSection(&mNative.lWaiters);
            if (waitDone)
            {
                break;
            }
        }
        mMutex->lock();
        EnterCriticalSection(&mNative.lWaiters);
        mNative.nWaiters--;
        mNative.nRelease--;
        bool lastWaiter = (mNative.nRelease == 0);
        LeaveCriticalSection(&mNative.lWaiters);
        if (lastWaiter)
        {
            ResetEvent(mNative.event);
        }
        return true;
    }
    else return wait();
}

bool sys::ConditionVarWin32::wait()
{
    dbg_printf("Waiting on condition\n");
    EnterCriticalSection(&mNative.lWaiters);

    mNative.nWaiters++;

    int myGeneration = mNative.nWaitGeneration;

    LeaveCriticalSection(&mNative.lWaiters);
    mMutex->unlock();

    for (;;)
    {
        WaitForSingleObject(mNative.event, INFINITE);
        EnterCriticalSection(&mNative.lWaiters);

        bool waitDone = mNative.nRelease > 0 &&
                        mNative.nWaitGeneration != myGeneration;
        LeaveCriticalSection(&mNative.lWaiters);
        if (waitDone)
        {
            break;
        }
    }
    mMutex->lock();
    EnterCriticalSection(&mNative.lWaiters);
    mNative.nWaiters--;
    mNative.nRelease--;
    bool lastWaiter = (mNative.nRelease == 0);
    LeaveCriticalSection(&mNative.lWaiters);
    if (lastWaiter)
    {
        ResetEvent(mNative.event);
    }
    return true;
}

bool sys::ConditionVarWin32::signal()
{
    dbg_printf("Signalling condition\n");
    EnterCriticalSection(&mNative.lWaiters);
    if (mNative.nWaiters > mNative.nRelease)
    {
        SetEvent(mNative.event);
        mNative.nRelease++;
        mNative.nWaitGeneration++;
    }
    LeaveCriticalSection(&mNative.lWaiters);
    return true;
}

bool sys::ConditionVarWin32::broadcast()
{
    dbg_printf("Broadcasting condition\n");
    EnterCriticalSection(&mNative.lWaiters);
    if (mNative.nWaiters > 0)
    {
        SetEvent(mNative.event);
        mNative.nRelease = mNative.nWaiters;
        mNative.nWaitGeneration++;
    }
    LeaveCriticalSection(&mNative.lWaiters);
    return true;
}

#endif // No other thread package

#endif // Windows

