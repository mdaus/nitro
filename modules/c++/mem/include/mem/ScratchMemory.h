/* =========================================================================
 * This file is part of mem-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2018, MDA Information Systems LLC
 *
 * mem-c++ is free software; you can redistribute it and/or modify
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

#ifndef __MEM_SCRATCHMEMORY_H__
#define __MEM_SCRATCHMEMORY_H__

#include <stddef.h>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <except/Exception.h>
#include <mem/BufferView.h>
#include <sys/Conf.h>

namespace mem
{

/*!
 *  \class ScratchMemory
 *  \brief Handle reservation of scratch memory segments within a single buffer.
 *
 *  Memory segments should be reserved during a "setup" phase using the put
 *  method. Once all segments are reserved, call the setup method to set up
 *  the underlying memory and ensure the alignment requirements of each segment.
 *  The get method may be used afterwards to obtain pointers to the memory
 *  segments.
 */
class ScratchMemory
{
public:
    ScratchMemory() :
        mNumBytesNeeded(0),
        mIsSetup(false)
    {
    }

    /*!
     * \brief Reserve a buffer segment within this scratch memory buffer.
     *
     * \param key Identifier for scratch segment
     * \param numBytes Size of scratch buffer
     * \param numBuffers Number of distinct buffers to set up
     * \param alignment Number of bytes to align segment pointer
     *
     * \throws except::Exception if the given key has already been used
     */
    template <typename T>
    void put(const std::string& key,
             size_t numBytes,
             size_t numBuffers = 1,
             size_t alignment = sys::SSE_INSTRUCTION_ALIGNMENT);

    /*!
     * \brief Get pointer to buffer segment.
     *
     * \param key Identifier for scratch segment
     * \param indexBuffer Index of distinct buffer
     *
     * \return Pointer to buffer segment
     *
     * \throws except::Exception if the scratch memory has not been set up,
     *         the key does not exist, or index of buffer is out of bounds
     */
    template <typename T>
    T* get(const std::string& key, size_t indexBuffer = 0);

    /*!
     * \brief Ensure underlying memory is properly set up and position segment
     *        pointers.
     *
     * \param scratchBuffer Storage to use for scratch memory. If scratchBuffer
     *        of size 0 is passed, memory is allocated internally.
     *
     * \throws except::Exception if the scratchBuffer passed in is too small
     *         to hold the requested scratch memory or has size > 0 with null
     *         data pointer
     */
    void setup(const mem::BufferView<sys::ubyte>& scratchBuffer =
            mem::BufferView<sys::ubyte>());

    /*!
     * \brief Get number of bytes needed to store scratch memory, including the
     *        maximum possible alignment overhead.
     */
    size_t getNumBytes() const
    {
        return mNumBytesNeeded;
    }

private:
    ScratchMemory(const ScratchMemory&);
    ScratchMemory& operator=(const ScratchMemory&);

    struct Segment
    {
        Segment(size_t numBytes, size_t numBuffers, size_t alignment) :
            numBytes(numBytes),
            numBuffers(numBuffers),
            alignment(alignment)
        {
        }
        size_t numBytes;
        size_t numBuffers;
        size_t alignment;
        std::vector<sys::ubyte*> buffers;
    };

    std::map<std::string, Segment> mSegments;
    std::vector<sys::ubyte> mStorage;
    mem::BufferView<sys::ubyte> mBuffer;
    size_t mNumBytesNeeded;
    bool mIsSetup;
};

}

#include <mem/ScratchMemory.hpp>

#endif
