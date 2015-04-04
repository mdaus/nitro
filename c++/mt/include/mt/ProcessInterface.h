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


#ifndef __MT_RUNTIME_PROCESS_INTERFACE_H__
#define __MT_RUNTIME_PROCESS_INTERFACE_H__


/*!
 *  \file
 *  \brief Runtime, system-independent process creation API
 *
 *  When it comes to multitasking, we almost all prefer threads to
 *  heritage process calls.  However, threads and processes are almost
 *  never equivalent.  Sometimes we need a process.  Here we define
 *  a simple API for process creation in a system-independent manner
 *
 */

#include "str/Dbg.h"
#include "mt/Runnable.h"
#include <vector>

namespace mt
{

template <typename Pid_T> class ProcessInterface : public mt::Runnable
{
public:
    enum { THE_CHILD = 0 };
    enum { PROCESS_CREATE_FAILED = -1 };
    ProcessInterface()
    {
        mTarget = this;
    }
    ProcessInterface(mt::Runnable* target) : mTarget(target)
    {}

    virtual ~ProcessInterface()
    {}

    virtual void start() = 0;
    virtual void waitFor() = 0;
    virtual void run() = 0;

protected:
    Pid_T mChildProcessID;
    mt::Runnable* mTarget;
};
}

#endif
