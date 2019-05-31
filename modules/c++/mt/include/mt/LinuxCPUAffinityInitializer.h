/* =========================================================================
 * This file is part of mt-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2019, MDA Information Systems LLC
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


#ifndef __MT_LINUX_CPU_AFFINITY_INITIALIZER_H__
#define __MT_LINUX_CPU_AFFINITY_INITIALIZER_H__

#if !defined(__APPLE_CC__)
#if defined(__linux) || defined(__linux__)

#include <memory>
#include <vector>

#include <sys/ScopedCPUAffinityUnix.h>
#include <mt/AbstractCPUAffinityInitializer.h>
#include <mt/LinuxCPUAffinityThreadInitializer.h>

namespace mt
{
class LinuxCPUAffinityInitializer : public AbstractCPUAffinityInitializer
{
public:
    LinuxCPUAffinityInitializer();

    virtual LinuxCPUAffinityThreadInitializer* newThreadInitializer()
    {
        return new LinuxCPUAffinityThreadInitializer(nextCPU());
    }

private:
    std::auto_ptr<const sys::ScopedCPUMaskUnix> nextCPU();

    const std::vector<int> mCPUs;
    size_t mNextCPUIndex;
};
}

#endif
#endif
#endif

