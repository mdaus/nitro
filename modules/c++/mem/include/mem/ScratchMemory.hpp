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

namespace mem
{
template <typename T>
void ScratchMemory::put(const std::string& key,
                        size_t numElements,
                        size_t numBuffers,
                        size_t alignment)
{
    put<sys::ubyte>(key, numElements * sizeof(T), numBuffers, alignment);
}

template <>
inline void ScratchMemory::put<sys::ubyte>(const std::string& key,
                                           size_t numElements,
                                           size_t numBuffers,
                                           size_t alignment)
{
    // invalidate buffer (setup must be called before any subsequent get call)
    mBuffer.data = NULL;

    size_t segmentOffset = mOffset;

    alignment = std::max<size_t>(1, alignment);
    mOffset += numBuffers * (numElements + alignment - 1);

    mNumBytesNeeded = std::max<size_t>(mNumBytesNeeded, mOffset);

    std::map<std::string, Segment>::iterator iterSeg = mSegments.find(key);
    if (iterSeg != mSegments.end())
    {
        std::ostringstream oss;
        oss << "Scratch memory space was already reserved for " << key;
        throw except::Exception(Ctxt(oss.str()));
    }
    mSegments.insert(
            iterSeg,
            std::make_pair(key, Segment(numElements, numBuffers, alignment, segmentOffset)));

    mKeyOrder.push_back(key);
}

inline void ScratchMemory::release(const std::string& key)
{
    std::map<std::string, Segment>::const_iterator iterSeg = mSegments.find(key);
    mReleasedKeys.insert(key);

    if (mKeyOrder.back() == key)
    {
        const Segment& segment = iterSeg->second;
        mOffset = segment.offset;
    }
    else
    {
        const Segment& segment = iterSeg->second;
        mOffset = segment.offset;

        std::vector<std::string>::iterator keyIter = std::find(mKeyOrder.begin(),
                                                                     mKeyOrder.end(),
                                                                     key);
        std::vector<std::string>::iterator nextKeyIter = mKeyOrder.erase(keyIter);
        mKeyOrder.push_back(key);

        bool keepGoing = true;
        std::string firstReleasedKey;
        bool firstReleasedKeyFound = false;

        while (keepGoing)
        {
            if (*nextKeyIter == key)
            {
                keepGoing = false;
            }

            //Get data for the segment that will be moved
            std::map<std::string, Segment>::const_iterator mapIter = 
                    mSegments.find(*nextKeyIter);
            const Segment& segmentToBeMoved = mapIter->second;

            const size_t numElements = segmentToBeMoved.numBytes;
            const size_t numBuffers = segmentToBeMoved.numBuffers;
            const size_t alignment = segmentToBeMoved.alignment;

            mSegments.erase(*nextKeyIter);
            std::string keyToInsert = *nextKeyIter;
            nextKeyIter = mKeyOrder.erase(nextKeyIter);

            if (mReleasedKeys.find(keyToInsert) != mReleasedKeys.end())
            {
                if (!firstReleasedKeyFound)
                {
                    firstReleasedKey = keyToInsert;
                    firstReleasedKeyFound = true;
                }
            }
            else
            {
                if (firstReleasedKeyFound)
                {
                    std::map<std::string, Segment>::const_iterator iterSegNew =
                            mSegments.find(firstReleasedKey);
                    const Segment& segmentNew = iterSegNew->second;
                    mOffset = segmentNew.offset;
                }
                firstReleasedKeyFound = false;
            }
            put<sys::ubyte>(keyToInsert, numElements, numBuffers, alignment);

        }
        std::map<std::string, Segment>::const_iterator iterSegNew = 
                mSegments.find(firstReleasedKey);
        const Segment& segmentNew = iterSegNew->second;
        mOffset = segmentNew.offset;
    }
}

template <typename T>
T* ScratchMemory::get(const std::string& key, size_t indexBuffer)
{
    return reinterpret_cast<T*>(
            lookupSegment(key, indexBuffer).buffers[indexBuffer]);
}

template <typename T>
const T* ScratchMemory::get(const std::string& key, size_t indexBuffer) const
{
    return reinterpret_cast<const T*>(
            lookupSegment(key, indexBuffer).buffers[indexBuffer]);
}

template <typename T>
BufferView<T> ScratchMemory::getBufferView(const std::string& key,
                                           size_t indexBuffer)
{
    const Segment& segment = lookupSegment(key, indexBuffer);
    return BufferView<T>(reinterpret_cast<T*>(segment.buffers[indexBuffer]),
                         segment.numBytes);
}

template <typename T>
BufferView<const T> ScratchMemory::getBufferView(const std::string& key,
                                                 size_t indexBuffer) const
{
    const Segment& segment = lookupSegment(key, indexBuffer);
    return BufferView<const T>(
            reinterpret_cast<const T*>(segment.buffers[indexBuffer]),
            segment.numBytes);
}
}
