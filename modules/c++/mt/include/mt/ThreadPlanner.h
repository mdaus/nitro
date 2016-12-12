/* =========================================================================
 * This file is part of mt-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2014, MDA Information Systems LLC
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

#ifndef __MT_THREAD_PLANNER_H__
#define __MT_THREAD_PLANNER_H__

#include <stddef.h>

namespace mt
{
/*!
 * \class ThreadPlanner
 * \brief Assists with dividing up work evenly between threads
 */
class ThreadPlanner
{
public:
    /*!
     * Constructor
     *
     * \param numElements The total number of elements of work to be divided
     * among threads
     * \param numThreads The number of threads that will be used for the work
     */
    ThreadPlanner(size_t numElements, size_t numThreads);

    size_t getNumElementsPerThread() const
    {
        return mNumElementsPerThread;
    }

    bool getThreadInfo(size_t threadNum,
                       size_t& startElement,
                       size_t& numElementsThisThread) const;

    // TODO: Give an example of when this will occur
    // TODO: This isn't right if there is 1 element and > 1 thread
    //       Should the math be floor(numElements / numElementsPerThread)?
    size_t getNumThreadsThatWillBeUsed() const;

private:
   size_t mNumElements;
   size_t mNumThreads;
   size_t mNumElementsPerThread;
};
}

#endif

