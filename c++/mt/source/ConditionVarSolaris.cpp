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


#if defined(__sun)
#include <thread.h>
#include <synch.h>
#include "mt/ConditionVarSolaris.h"

mt::ConditionVarSolaris::ConditionVarSolaris() :
    mMutexOwned(new mt::MutexSolaris()),
    mMutex(mMutexOwned.get())
{
    if ( ::cond_init(&mNative, NULL, NULL) != 0)
        throw mt::ThreadResourceException("ConditionVar initialization failed");
}

mt::ConditionVarSolaris::ConditionVarSolaris(mt::MutexSolaris* theLock, bool isOwner) :
    mMutex(theLock)
{
    if (!theLock)
        throw except::NullPointerReference("ConditionVar received NULL mutex");

    if (isOwner)
        mMutexOwned.reset(theLock);

    if ( ::cond_init(&mNative, NULL, NULL) != 0)
        throw mt::ThreadResourceException("ConditionVar initialization failed");
}

mt::ConditionVarSolaris::~ConditionVarSolaris()
{
    ::cond_destroy(&mNative);
}

void mt::ConditionVarSolaris::acquireLock()
{
    mMutex->lock();
}

void mt::ConditionVarSolaris::dropLock()
{
    mMutex->unlock();
}

void mt::ConditionVarSolaris::signal()
{
    dbg_printf("Signaling condition\n");
    if (::cond_signal(&mNative) != 0)
        throw mt::ConditionVarException("ConditionVar signal failed");
}

void mt::ConditionVarSolaris::wait()
{
    dbg_printf("Waiting on condition\n");
    if (::cond_wait(&mNative, &(mMutex->getNative())) != 0)
        throw mt::ConditionVarException("ConditionVar wait failed");
}

void mt::ConditionVarSolaris::wait(double seconds)
{
    dbg_printf("Timed waiting on condition [%f]\n", seconds);
    if ( seconds > 0 )
    {
        timestruc_t tout;
        tout.tv_sec = time(NULL) + (int)seconds;
        tout.tv_nsec = (int)((seconds - (int)(seconds)) * 1e9);
        if (::cond_timedwait(&mNative,
                             &(mMutex->getNative()),
                             &tout) != 0)
            throw mt::ConditionVarException("ConditionVar wait failed");
    }
    else
        wait();
}

void mt::ConditionVarSolaris::broadcast()
{
    dbg_printf("Broadcasting condition\n");
    if (::cond_broadcast(&mNative) != 0)
        throw mt::ConditionVarException("ConditionVar broadcast failed");
}

cond_t& mt::ConditionVarSolaris::getNative()
{
    return mNative;
}

#endif

