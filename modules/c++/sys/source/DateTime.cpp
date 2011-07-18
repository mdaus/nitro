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

#include <string.h>
#include <errno.h>
#include <sstream>
#include <iomanip>

#include "sys/Conf.h"
#include "except/Exception.h"
#include "sys/DateTime.h"
#include "str/Convert.h"

#if defined(HAVE_SYS_TIME_H)
#include <sys/time.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

sys::DateTime::DateTime()
{
    setNow();
}

void sys::DateTime::setNow()
{
#if defined(__POSIX) && defined(USE_CLOCK_GETTIME)
    struct timespec now;
    clock_gettime(CLOCK_REALTIME,&now);
    mTimeInMillis = (now.tv_sec + 1.0e-9 * now.tv_nsec) * 1000;
#elif defined(HAVE_SYS_TIME_H)
    struct timeval now;
    gettimeofday(&now,NULL);
    mTimeInMillis = (now.tv_sec + 1.0e-6 * now.tv_usec) * 1000;
#elif defined(_WIN32)
    // Getting time twice may be inefficient but is quicker
    // than converting the SYSTEMTIME structure into
    // milliseconds
    // We could add an additional flag here if the user
    // does not need millisecond accuracy
    SYSTEMTIME now;
    GetLocalTime(&now);
    mTimeInMillis = (double)time(NULL) * 1000 + now.wMilliseconds;
#else
    mTimeInMillis = (double)time(NULL) * 1000;
#endif
    fromMillis();
}

void sys::DateTime::getLocalTime(tm &localTime) const
{
    getLocalTime(static_cast<time_t>(mTimeInMillis / 1000), localTime);
}

void sys::DateTime::getLocalTime(time_t numSecondsSinceEpoch, tm &localTime)
{
    // Would like to use the reentrant version.  If we don't have one, cross
    // our fingers and hope the regular function actually is reentrant
    // (supposedly this is the case on Windows).
#ifdef HAVE_LOCALTIME_R
    if (::localtime_r(&numSecondsSinceEpoch, &localTime) == NULL)
    {
        int const errnum = errno;
        throw except::Exception(Ctxt("localtime_r() failed (" +
            std::string(::strerror(errnum)) + ")"));
    }
#else
    tm const * const localTimePtr = ::localtime(&numSecondsSinceEpoch);
    if (localTimePtr == NULL)
    {
        int const errnum = errno;
        throw except::Exception(Ctxt("localtime failed (" +
            std::string(::strerror(errnum)) + ")"));
    }
    localTime = *localTimePtr;
#endif
}

void sys::DateTime::getGMTime(tm &gmTime) const
{
    getGMTime(static_cast<time_t>(mTimeInMillis / 1000), gmTime);
}

void sys::DateTime::getGMTime(time_t numSecondsSinceEpoch, tm &gmTime)
{
    // Would like to use the reentrant version.  If we don't have one, cross
    // our fingers and hope the regular function actually is reentrant
    // (supposedly this is the case on Windows).
#ifdef HAVE_GMTIME_R
    if (::gmtime_r(&numSecondsSinceEpoch, &gmTime) == NULL)
    {
        int const errnum = errno;
        throw except::Exception(Ctxt("gmtime_r() failed (" +
            std::string(::strerror(errnum)) + ")"));
    }
#else
    tm const * const gmTimePtr = ::gmtime(&numSecondsSinceEpoch);
    if (gmTimePtr == NULL)
    {
        int const errnum = errno;
        throw except::Exception(Ctxt("gmtime failed (" +
            std::string(::strerror(errnum)) + ")"));
    }
    gmTime = *gmTimePtr;
#endif
}

void sys::DateTime::fromMillis()
{
    tm t;
    getLocalTime(t);

    // this is year since 1900 so need to add that
    mYear = t.tm_year + 1900;
    // 0-based so add 1
    mMonth = t.tm_mon + 1;
    mDayOfMonth = t.tm_mday;
    mDayOfWeek = t.tm_wday + 1;
    mDayOfYear = t.tm_yday + 1;
    mHour = t.tm_hour;
    mMinute = t.tm_min;
    mDST = t.tm_isdst;

    const double timediff = (mTimeInMillis / 1000) - mktime(&t);
    mSecond = t.tm_sec + timediff;
}

void sys::DateTime::toMillis()
{
    tm t;
    t.tm_year = mYear - 1900;
    t.tm_mon = mMonth - 1;
    t.tm_mday = mDayOfMonth;
    t.tm_wday = mDayOfWeek + 1;
    t.tm_yday = mDayOfYear - 1;
    t.tm_hour = mHour;
    t.tm_min = mMinute;
    t.tm_sec = (int)mSecond;
    t.tm_isdst = mDST;

    mTimeInMillis = toMillis(t);
}

double sys::DateTime::toMillis(tm t) const
{
    time_t timeInSeconds = mktime(&t);
    double timediff = mSecond - t.tm_sec;
    return (timeInSeconds + timediff) * 1000;
}

sys::DateTime::DateTime(int hour, int minute, double second)
{
    setNow();

    mHour = hour;
    mMinute = minute;
    mSecond = second;

    toMillis();
}

sys::DateTime::DateTime(int year, int month, int day)
{
    mYear = year;
    mMonth = month;
    mDayOfMonth = day;
    mHour = 0;
    mMinute = 0;
    mSecond = 0;

    toMillis();
    fromMillis();
}

sys::DateTime::DateTime(int year, int month, int day,
			int hour, int minute, double second)
{
    mYear = year;
    mMonth = month;
    mDayOfMonth = day;
    mHour = hour;
    mMinute = minute;
    mSecond = second;

    toMillis();
    fromMillis();
}

sys::DateTime::DateTime(double timeInMillis)
{
    mTimeInMillis = timeInMillis;
    fromMillis();
}

double sys::DateTime::getGMTimeInMillis() const
{
    tm gmTime;
    getGMTime(gmTime);
    gmTime.tm_isdst = mDST;
    return toMillis(gmTime);
}

void sys::DateTime::setDayOfMonth(int dayOfMonth)
{
    mDayOfMonth = dayOfMonth;
    toMillis();
}

void sys::DateTime::setMonth(int month)
{
    mMonth = month;
    toMillis();
}

void sys::DateTime::setHour(int hour)
{
    mHour = hour;
    toMillis();
}

void sys::DateTime::setMinute(int minute)
{
    mMinute = minute;
    toMillis();
}

void sys::DateTime::setSecond(double second)
{
    mSecond = second;
    toMillis();
}

void sys::DateTime::setTimeInMillis(double time)
{
    mTimeInMillis = time;
    fromMillis();
}

void sys::DateTime::setYear(int year)
{
    mYear = year;
    toMillis();
}

void sys::DateTime::setDST(bool isDST)
{
    if(isDST)
	mDST = 1;
    else
	mDST = 0;
}

void sys::DateTime::format(FormatTypes formatType,
                           std::string& formatStr) const
{
    std::ostringstream ostr;
    ostr.fill('0');

    switch (formatType)
    {
    case FORMAT_ISO_8601:
    {
        // We're in local time but this format is in GMT, so need to convert
        DateTime const gmDT(getGMTimeInMillis());

        ostr << std::setw(4) << gmDT.mYear << "-"
             << std::setw(2) << gmDT.mMonth << "-"
             << std::setw(2) << gmDT.mDayOfMonth << "T"
             << std::setw(2) << gmDT.mHour << ":"
             << std::setw(2) << gmDT.mMinute << ":"
             << std::setw(2) << static_cast<int>(gmDT.mSecond) << "Z";
    }
        break;

    default:
        throw except::Exception(Ctxt("Unknown format type " +
                                         str::toString(formatType)));
    }

    formatStr = ostr.str();
}
