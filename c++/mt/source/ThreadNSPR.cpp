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


#  if defined(USE_NSPR_THREADS)
#include "mt/ThreadNSPR.h"
void mt::ThreadNSPR::__start(void *v)
{
    STANDARD_START_CALL(ThreadNSPR, v);
}
void mt::ThreadNSPR::start()
{

    PRThreadType type = (getLevel() == mt::ThreadNSPR::KERNEL_LEVEL) ?
                        (PR_SYSTEM_THREAD) : (PR_USER_THREAD);

    PRThreadScope scope = (mIsLocal) ? (PR_LOCAL_THREAD) :
                          (PR_GLOBAL_THREAD);

    PRThreadPriority priority;
    if (getPriority() == mt::ThreadNSPR::NORMAL_PRIORITY)
        priority = PR_PRIORITY_NORMAL;
    else if (getPriority() == mt::ThreadNSPR::MAXIMUM_PRIORITY)
        priority = PR_PRIORITY_HIGH;
    else if (getPriority() == mt::ThreadNSPR::MINIMUM_PRIORITY)
        priority = PR_PRIORITY_LOW;
    mNative = PR_CreateThread(type,
                              (void (*)(void *))this->__start,
                              this,
                              priority,
                              scope,
                              PR_JOINABLE_THREAD,
                              0);
}

void mt::ThreadNSPR::join()
{
    if (!PR_JoinThread(mNative))
        throw mt::ThreadResourceException("join()");
}

void mt::ThreadNSPR::yield()
{
    PR_Sleep(PR_INTERVAL_NO_WAIT);
}

#  endif

