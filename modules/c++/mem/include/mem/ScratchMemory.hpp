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
                        size_t numBytes,
                        size_t numBuffers,
                        size_t alignment)
{
    put<sys::ubyte>(key, numBytes * sizeof(T), numBuffers, alignment);
}

template <>
inline void ScratchMemory::put<sys::ubyte>(const std::string& key,
                                           size_t numBytes,
                                           size_t numBuffers,
                                           size_t alignment)
{
    alignment = std::max<size_t>(1, alignment);
    mNumBytesNeeded += numBuffers * (numBytes + alignment - 1);

    std::map<std::string, Segment>::iterator iterSeg = mSegments.find(key);
    if (iterSeg != mSegments.end())
    {
        std::ostringstream oss;
        oss << "Scratch memory space was already reserved for " << key;
        throw except::Exception(Ctxt(oss.str()));
    }
    mSegments.insert(
            iterSeg,
            std::make_pair(key, Segment(numBytes, numBuffers, alignment)));
}

template <typename T>
T* ScratchMemory::get(const std::string& key, size_t indexBuffer)
{
    if (!mIsSetup)
    {
        std::ostringstream oss;
        oss << "Tried to get scratch memory for \"" << key
            << "\" before running setup.";
        throw except::Exception(Ctxt(oss.str()));
    }

    std::map<std::string, Segment>::iterator iterSeg = mSegments.find(key);
    if (iterSeg == mSegments.end())
    {
        std::ostringstream oss;
        oss << "Scratch memory segment was not found for \"" << key << "\"";
        throw except::Exception(Ctxt(oss.str()));
    }

    Segment& segment = iterSeg->second;
    if (indexBuffer >= segment.buffers.size())
    {
        std::ostringstream oss;
        oss << "Trying to get buffer index " << indexBuffer << " for \""
            << key << "\", which has only " << segment.buffers.size()
            << " buffers";
        throw except::Exception(Ctxt(oss.str()));
    }
    return reinterpret_cast<T*>(segment.buffers[indexBuffer]);
}

}
