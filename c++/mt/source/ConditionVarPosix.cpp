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

#include <mt/ConditionVarPosix.h>

#if defined(HAVE_PTHREAD_H)

#include <pthread.h>

mt::ConditionVarPosix::ConditionVarPosix() :
    mMutexOwned(new mt::MutexPosix()),
    mMutex(mMutexOwned.get())
{
    if ( ::pthread_cond_init(&mNative, NULL) != 0)
        throw ThreadResourceException("ConditionVar initialization failed");
}

mt::ConditionVarPosix::ConditionVarPosix(mt::MutexPosix* theLock, bool isOwner) :
    mMutex(theLock)
{
    if (!theLock)
        throw except::NullPointerReference("ConditionVar received NULL mutex");

    if (isOwner)
        mMutexOwned.reset(theLock);

    if ( ::pthread_cond_init(&mNative, NULL) != 0)
        throw ThreadResourceException("ConditionVar initialization failed");
}

mt::ConditionVarPosix::~ConditionVarPosix()
{
    ::pthread_cond_destroy(&mNative);
}

void mt::ConditionVarPosix::acquireLock()
{
    mMutex->lock();
}

void mt::ConditionVarPosix::dropLock()
{
    mMutex->unlock();
}

void mt::ConditionVarPosix::signal()
{
    dbg_printf("Signaling condition\n");
    if (::pthread_cond_signal(&mNative) != 0)
        throw mt::ConditionVarException("ConditionVar signal failed");
}

void mt::ConditionVarPosix::wait()
{
    dbg_printf("Waiting on condition\n");
    if (::pthread_cond_wait(&mNative, &(mMutex->getNative())) != 0)
        throw mt::ConditionVarException("ConditionVar wait failed");
}

void mt::ConditionVarPosix::wait(double seconds)
{
    dbg_printf("Timed waiting on condition [%f]\n", seconds);

    if ( seconds > 0 )
    {
        timespec tout;
        tout.tv_sec = time(NULL) + (int)seconds;
        tout.tv_nsec = (int)((seconds - (int)(seconds)) * 1e9);
        if (::pthread_cond_timedwait(&mNative,
                                     &(mMutex->getNative()),
                                     &tout) != 0)
            throw mt::ConditionVarException("ConditionVar wait failed");
    }
    else
        wait();
}

void mt::ConditionVarPosix::broadcast()
{
    dbg_printf("Broadcasting condition\n");
    if (::pthread_cond_broadcast(&mNative) != 0)
        throw mt::ConditionVarException("ConditionVar broadcast failed");
}

pthread_cond_t& mt::ConditionVarPosix::getNative()
{
    return mNative;
}

#endif
