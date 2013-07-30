/* =========================================================================
 * This file is part of io-c++ 
 * =========================================================================
 * 
 * (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
 *
 * io-c++ is free software; you can redistribute it and/or modify
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

#include "io/ByteStream.h"

sys::Off_T io::ByteStream::available()
{
    sys::Off_T where = mPosition;
    sys::Off_T until = static_cast<sys::Off_T>(mData.size());
    sys::Off_T diff = until - where;
    return (diff < 0) ? 0 : diff;
}

void io::ByteStream::write(const sys::byte *b, sys::Size_T size)
{
    // append the data to the end of the vector
    size_t writePosition(mData.size());
    size_t newSize(mData.size() + size);
    mData.resize(newSize);
    std::copy(b, b+size, &mData[writePosition]);
}

sys::SSize_T io::ByteStream::read(sys::byte *b, sys::Size_T len)
{
    sys::Off_T maxSize = available();
    if (maxSize <= 0) return io::InputStream::IS_END;

    if (maxSize <  static_cast<sys::Off_T>(len)) len = maxSize;
    if (len     <= 0)                            return 0;

    memcpy(b, &mData[mPosition], len);
    mPosition += len;
    return len;
}

