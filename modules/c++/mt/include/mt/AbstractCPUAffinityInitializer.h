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


#ifndef __MT_ABSTRACT_CPU_AFFINITY_INITIALIZER_H__
#define __MT_ABSTRACT_CPU_AFFINITY_INITIALIZER_H__

#include <mt/AbstractCPUAffinityThreadInitializer.h>

namespace mt
{
/*!
 * \class AbstractCPUAffinityInitializer
 * \brief Abstract class for providing thread-level affinity initializers.
 *        Should be used by the primary thread to generate data for
 *        child threads.
 */
class AbstractCPUAffinityInitializer
{
public:
    AbstractCPUAffinityInitializer() {}
    virtual ~AbstractCPUAffinityInitializer() {}

    /*!
     * \returns a new thread initializer. In general, this should return
     *          a different affinity initializer each time it is called.
     */
    virtual AbstractCPUAffinityThreadInitializer* newThreadInitializer() = 0;
};
}

#endif
