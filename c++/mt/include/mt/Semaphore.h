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


#ifndef __MT_SEMAPHORE_H__
#define __MT_SEMAPHORE_H__
/**
 *  \file 
 *  \brief Include the right semaphore.
 *
 *  This file will auto-select the semaphore of choice,
 *  if one is to be defined.  
 *  \note We need to change the windows part to check _MT
 *  because that is how it determines reentrance!
 *
 */

#    if defined(USE_NSPR_THREADS)
#        include "mt/SemaphoreNSPR.h"
namespace mt
{
typedef SemaphoreNSPR Semaphore;
}
#    elif defined(WIN32)
#        include "mt/SemaphoreWin32.h"
namespace mt
{
typedef SemaphoreWin32 Semaphore;
}
#    elif defined(__sun)
#        include "mt/SemaphoreSolaris.h"
namespace mt
{
typedef SemaphoreSolaris Semaphore;
}
#    elif defined(__sgi)
#        include "mt/SemaphoreIrix.h"
namespace mt
{
typedef SemaphoreIrix Semaphore;
}
#    elif defined(__APPLE_CC__)
typedef int Semaphore;
// Give 'em Posix
#    else
#        include "mt/SemaphorePosix.h"
namespace mt
{
typedef SemaphorePosix Semaphore;
}
#    endif // Which thread package?

#endif // End of header
