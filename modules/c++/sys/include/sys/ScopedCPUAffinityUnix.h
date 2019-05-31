/* =========================================================================
 * This file is part of sys-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2019, MDA Information Systems LLC
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

#ifndef __SYS_SCOPED_CPU_AFFINITY_UNIX_H__
#define __SYS_SCOPED_CPU_AFFINITY_UNIX_H__

#include "sys/sys_config.h"

#if !defined(WIN32)

#include <memory>
#include <sched.h>

namespace sys
{
class ScopedCPUMaskUnix
{
public:
    ScopedCPUMaskUnix();
    ScopedCPUMaskUnix(int numCPUs);
    ~ScopedCPUMaskUnix();

    //! \returns the CPU set
    const cpu_set_t* getMask() const
    {
        return mMask;
    }

    cpu_set_t* getMask()
    {
        return mMask;
    }

    //! \returns the size of the CPU set in bytes
    size_t getSize() const
    {
        return mSize;
    }

private:

    void initialize(int numCPUs);

    size_t mSize;
    cpu_set_t* mMask;
};

class ScopedCPUAffinityUnix
{
public:
    ScopedCPUAffinityUnix();

    //! \returns the CPU set represeting the affinity mask
    const cpu_set_t* getMask() const
    {
        return mCPUMask->getMask();
    }

    //! \returns the size of the CPU set in bytes
    size_t getSize() const
    {
        return mCPUMask->getSize();
    }

    //! \returns the number of online CPUs
    static int getNumOnlineCPUs();

private:
    std::auto_ptr<const ScopedCPUMaskUnix> mCPUMask;
};
}

#endif
#endif
