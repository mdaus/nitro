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


#if !defined(WIN32)
#include <time.h>
#include <string.h>
#include <errno.h>
#include "sys/TimeStamp.h"
#include "sys/Conf.h"
#include "except/Exception.h"

std::string sys::TimeStamp::local()
{
    // Get the current time
    time_t __time = ::time(0);
    if (__time == static_cast<time_t>(-1))
    {
        int const errnum = errno;
        throw except::Exception(Ctxt("time() failed (" +
            std::string(::strerror(errnum)) + ")"));
    }

    // Convert it to local time
    tm temp;
    if (::localtime_r(&__time, &temp) == NULL)
    {
        int const errnum = errno;
        throw except::Exception(Ctxt("localtime_r() failed (" +
            std::string(::strerror(errnum)) + ")"));
    }

    // Format it
    char timeStamp[MAX_TIME_STAMP];
    ::strftime(timeStamp, MAX_TIME_STAMP - 1,
               getFormat(), &temp);
    return timeStamp;
}
std::string sys::TimeStamp::gmt()
{
    // Get the current time
    time_t __time = ::time(0);
    if (__time == static_cast<time_t>(-1))
    {
        int const errnum = errno;
        throw except::Exception(Ctxt("time() failed (" +
            std::string(::strerror(errnum)) + ")"));
    }

    // Convert it to GMT time
    tm temp;
    if (::gmtime_r(&__time, &temp) == NULL)
    {
        int const errnum = errno;
        throw except::Exception(Ctxt("gmtime_r() failed (" +
            std::string(::strerror(errnum)) + ")"));
    }

    // Format it
    char timeStamp[MAX_TIME_STAMP];
    ::strftime(timeStamp, MAX_TIME_STAMP - 1,
               getFormat(), &temp);
    return timeStamp;
}
const char* sys::TimeStamp::getFormat()
{
    if (m24HourTime)
    {
        return "%m/%d/%Y, %H:%M:%S";
    }
    return "%m/%d/%Y, %I:%M:%S%p";
}
#endif

