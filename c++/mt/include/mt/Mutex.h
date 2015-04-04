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


#ifndef __MT_MUTEX_H__
#define __MT_MUTEX_H__ 
/**
 *  \file 
 *  \brief Include the right mutex.
 *
 *  This file will auto-select the mutex of choice,
 *  if one is to be defined.  
 *  \note We need to change the windows part to check _MT
 *  because that is how it determines reentrance!
 *
 */

#    if defined(USE_NSPR_THREADS)
#        include "mt/MutexNSPR.h"
namespace mt
{
typedef MutexNSPR Mutex;
}
#    elif defined(WIN32)
#        include "mt/MutexWin32.h"
namespace mt
{
typedef MutexWin32 Mutex;
}

/* #    elif defined(USE_BOOST) */
/* #        include "MutexBoost.h" */
/*          typedef MutexBoost Mutex; */
#    elif defined(__sun)
#        include "mt/MutexSolaris.h"
namespace mt
{
typedef MutexSolaris Mutex;
}
#    elif defined(__sgi)
#        include "mt/MutexIrix.h"
namespace mt
{
typedef MutexIrix Mutex;
}
// Give 'em POSIX
#    else
#        include "mt/MutexPosix.h"
namespace mt
{
typedef MutexPosix Mutex;
}
#    endif // Which thread package?

#endif // End of header
