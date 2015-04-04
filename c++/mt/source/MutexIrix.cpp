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


#if defined(__sgi)
#include <ulocks.h>
#include "mt/MutexIrix.h"
#include "mt/SyncFactoryIrix.h"

mt::MutexIrix::MutexIrix()
{
    if (!mt::SyncFactoryIrix().createLock(*this))
        throw ThreadResourceException("Mutex initialization failed");
}

mt::MutexIrix::~MutexIrix()
{
    dbg_printf("~MutexIrix()\n");
    mt::SyncFactoryIrix().destroyLock(*this);
}

void mt::MutexIrix::lock()
{
    dbg_printf("MutexIrix::lock()\n");
    if (!mt::SyncFactoryIrix().setLock(*this))
        throw mt::LockException("Mutex lock failed");
}

void mt::MutexIrix::unlock()
{
    dbg_printf("MutexIrix::unlock()\n");
    if (!mt::SyncFactoryIrix().unsetLock(*this))
        throw mt::LockException("Mutex unlock failed");
}

ulock_t*& mt::MutexIrix::getNative()
{
    return mNative;
}

#endif // __sgi

