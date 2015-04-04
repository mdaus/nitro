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


#ifndef __MT_PROCESS_UNIX_H__
#define __MT_PROCESS_UNIX_H__


#if defined(__GNUC__)
#    if defined(__sgi) || defined(__sgi__)
#        ifdef _FIX_BROKEN_HEADERS
            typedef long long __int64_t;
            typedef __int64_t jid_t;
#        endif
#    endif
#endif

#if !defined(WIN32)

#include <assert.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "sys/Conf.h"
#include "mt/ProcessInterface.h"


namespace mt
{
class ProcessUnix : public ProcessInterface< sys::Pid_T >
{
public:
    ProcessUnix()
    {}
    ProcessUnix(Runnable* target): ProcessInterface< sys::Pid_T >(target)
    {}
    virtual ~ProcessUnix()
    {}
    void start();
    void waitFor();
};
}


#endif
#endif
