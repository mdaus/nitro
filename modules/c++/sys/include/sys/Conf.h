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


#ifndef __SYS_CONF_H__
#define __SYS_CONF_H__

#if defined (__APPLE_CC__)
#  include <iosfwd>
#endif

#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#if defined(__sgi) || defined(__sgi__)
#   include <stdarg.h>
#else
#   include <cstdarg>
#endif

#include <memory>
#include "str/Format.h"
#include "sys/TimeStamp.h"

#ifdef HAVE_CONFIG_H
#  include "sys/config.h"
#endif

/*  Dance around the compiler to figure out  */
/*  if we have access to function macro...   */

/*  If we have gnu -- Yes!!  */
#if defined(__GNUC__)
    /*  We get a really nice function macro  */
#   define NativeLayer_func__ __PRETTY_FUNCTION__
#elif defined(WIN32) && (_MSC_VER >= 1300)
#   define NativeLayer_func__ __FUNCSIG__
/*  Otherwise, lets look for C99 compatibility  */
#elif defined (__STDC_VERSION__)
    /*  The above line may not be necessary but   */
    /*  Im not about to find out...               */
    /*  Check to see if the compiler is doing C99 */
#   if __STDC_VERSION__ < 199901
        /*  If not, define an empty function signature  */
#       define NativeLayer_func__ ""
#   else
        /*  If so -- Yes!! Use it!  */
#       define NativeLayer_func__ __func__
#   endif
#else
/*  If STDC stuff isnt defined, that would be lame, but  */
/*  lets make sure its still okay                        */
#   define NativeLayer_func__ ""
#endif

#if defined(WIN32) || defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#      define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  include <process.h>

namespace sys
{
    typedef HANDLE            Handle_T;
    typedef char byte;
    typedef unsigned char     Uint8_T;
    typedef unsigned __int16  Uint16_T;
    typedef unsigned __int32  Uint32_T;
    typedef unsigned __int64  Uint64_T;
    typedef off_t             Off_T;
    typedef signed char       Int8_T;
    typedef __int16           Int16_T;
    typedef __int32           Int32_T;
    typedef __int64           Int64_T;
    typedef DWORD             Pid_T;
#   ifndef _SIZE_T_DEFINED
#       ifdef  _WIN64
            typedef unsigned __int64    Size_T;
#	else
	    typedef _W64 unsigned int   Size_T;
#       endif
#   else
        typedef size_t Size_T;
#   endif
#   ifdef  _WIN64
        typedef __int64    SSize_T;
#   else
        typedef _W64 int   SSize_T;
#   endif
}
#else // !windows
#   include <sys/types.h>
#   if defined(__sgi) || defined(__sgi__)
#       if defined(__GNUC__)
#           ifdef _FIX_BROKEN_HEADERS
                typedef __int64_t jid_t;
#           endif
#       endif
#   endif
#   if defined(__sun) || defined(__sun__) || defined(__sparc) || defined(__sparc) || defined(__sparc__)
#       if !defined(__SunOS_5_6) && !defined(__SunOS_5_7) && !defined(__SunOS_5_8) && defined(__GNUC__)
#           ifdef _FIX_BROKEN_HEADERS
                typedef id_t projid_t;
#           endif
#       endif
#       include <sys/stream.h>
#  endif
#  include <signal.h>
#  include <errno.h>
#  include <sys/stat.h>
#  include <sys/wait.h>
#  include <unistd.h>
#  include <fcntl.h>
#  include <dirent.h>
#  if defined(_USE_STDINT)
#      include <stdint.h>
#  else
#      include <inttypes.h>
#  endif
//#  include <sys/mman.h>

namespace sys
{
    
    typedef char               byte;
    typedef uint8_t            Uint8_T;
    typedef uint16_t           Uint16_T;
    typedef uint32_t           Uint32_T;
    typedef uint64_t           Uint64_T;
    typedef size_t             Size_T;
    typedef ssize_t            SSize_T;
    typedef off_t              Off_T;
    typedef int8_t             Int8_T;
    typedef int16_t            Int16_T;
    typedef int32_t            Int32_T;
    typedef int64_t            Int64_T;
    typedef int                Handle_T;
    // Should we remove this?
    typedef pid_t              Pid_T;
}
#endif // *nix

// For strerror
#include <string.h>

#ifdef signal //  Apache defines this
#    undef  signal
#endif


#define FmtX str::format

#define SYS_TIME sys::TimeStamp().local()

#define SYS_FUNC NativeLayer_func__

#define Ctxt(MESSAGE) except::Context(__FILE__, __LINE__, SYS_FUNC, SYS_TIME, MESSAGE)

namespace sys
{
    /*!
     * Returns true if the system is big-endian, otherwise false.
     * On Intel systems, we are usually small-endian, and on
     * RISC architectures we are big-endian.
     */
    bool isBigEndianSystem();

}

#endif

