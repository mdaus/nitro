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


#ifndef __MT_THREAD_H__
#define __MT_THREAD_H__ 

/**
 *  \file 
 *  \brief Include the right thread.
 *
 *  This file will auto-select the thread of choice,
 *  if one is to be defined.  
 *  \note We need to change the windows part to check _MT
 *  because that is how it determines reentrance!
 *
 */

// Netscape portable runtime
#    if defined(USE_NSPR_THREADS)
#        include "mt/ThreadNSPR.h"
namespace mt
{
typedef ThreadNSPR Thread;
}
#    elif defined(WIN32)
#        include "mt/ThreadWin32.h"
namespace mt
{
typedef ThreadWin32 Thread;
}
#    elif defined(__sun)
#        include "mt/ThreadSolaris.h"
namespace mt
{
typedef ThreadSolaris Thread;
}
#    elif defined(__sgi)
#        include "mt/ThreadIrix.h"
namespace mt
{
typedef ThreadIrix Thread;
}
// If they havent defined anything and its !windows, we'll give 'em posix
#    else
#        include "mt/ThreadPosix.h"
namespace mt
{
typedef ThreadPosix Thread;
}
#    endif // Which thread package?

#endif
