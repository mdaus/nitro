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

#include <mt/SemaphorePosix.h>

#if defined(HAVE_PTHREAD_H) && !defined(__APPLE_CC__)

#include <semaphore.h>

mt::SemaphorePosix::SemaphorePosix(unsigned int count)
{
    sem_init(&mNative, 0, count);
}

mt::SemaphorePosix::~SemaphorePosix()
{
    sem_destroy(&mNative);
}

void mt::SemaphorePosix::wait()
{
    if (sem_wait(&mNative) != 0)
        throw mt::ThreadResourceException("Semaphore wait failed");
}

void mt::SemaphorePosix::signal()
{
    if (sem_post(&mNative) != 0)
        throw mt::ThreadResourceException("Semaphore signal failed");
}

sem_t& mt::SemaphorePosix::getNative()
{
    return mNative;
}

#endif

