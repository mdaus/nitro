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


#ifndef __SYS_WIN32_CONDITION_VARIABLE_H__
#define __SYS_WIN32_CONDITION_VARIABLE_H__

#if defined(WIN32) && defined(_REENTRANT)
#if !defined(USE_NSPR_THREADS) && !defined(__POSIX)

#include "sys/ConditionVarInterface.h"
#include "sys/MutexWin32.h"


namespace sys
{
struct ConditionVarDataWin32
{
    ConditionVarDataWin32();
    ~ConditionVarDataWin32();
    int nWaiters;
    int nRelease;
    int nWaitGeneration;
    CRITICAL_SECTION lWaiters;
    HANDLE event;
};

class ConditionVarWin32 :
            public ConditionVarInterface < ConditionVarDataWin32,
            MutexWin32 >

{
public:
    typedef ConditionVarInterface < ConditionVarDataWin32,
    MutexWin32 > Parent_T;

    ConditionVarWin32()
    {
        mMutex = new MutexWin32();

    }
    ConditionVarWin32(MutexWin32 *theLock, bool isOwner = false) :
            Parent_T(theLock, isOwner)
    {}
    virtual ~ConditionVarWin32()
    {}
    /*!
     *  Wait for on a signal for a time interval.  
     *  This should eventually have
     *  a class TimeInterval as the second argument, which takes
     *  any time interval as a right-hand-side
     *  \param timeout How long to wait.  This is only temporarily
     *  a double
     *  \todo  Create a TimeInterval class, and use it as parameter
     *  \return true upon success
     */
    bool wait(double timeout);
    bool wait();
    bool signal();
    bool broadcast();

};
}
#endif

#endif

#endif
