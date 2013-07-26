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

#ifndef __IO_BYTE_STREAM_H__
#define __IO_BYTE_STREAM_H__

#include <vector>
#include "io/BidirectionalStream.h"
#include "sys/Conf.h"
#include "except/Error.h"
#include "except/Exception.h"
#include "io/SeekableStreams.h"

/*!
 *  \file
 *  \brief  Class for buffering data, inherits from
 *      SeekableBidirectionalStream
 *
 *  This type exists to handle piped data.  If all of your
 *  data is ascii, it is easy just to use a std::string from
 *  C++ to handle this.  However, for binary transfers, arbitrary
 *  0's can be anywhere (Null-bytes) making it impossible to use
 *  strings as containers.  
 * 
 *  Alternatively, we could have used std::stream<const char*>,
 *  but having it in a container makes it all less accessible, so we
 *  opted for our own growable data array
 */
namespace io
{
/*!
 *  \class ByteStream
 *  \brief  Class for buffering data
 *
 *  This type exists to handle piped data.  If all of your
 *  data is ascii, it is easy just to use a std::string from
 *  C++ to handle this.  However, for binary transfers, arbitrary
 *  0's can be anywhere (Null-bytes) making it impossible to use
 *  strings as containers.  
 */
class ByteStream : public SeekableBidirectionalStream
{
public:

    //! Default constructor
    ByteStream(sys::Size_T len = 0) :
        mData(len), mPosition(0)
    {
    }

    //! Destructor
    virtual ~ByteStream()
    {
    }

    sys::Off_T tell()
    {
        return mPosition;
    }

    sys::Off_T seek(sys::Off_T offset, Whence whence)
    {
        switch (whence)
        {
        case START:
            mPosition = offset;
            break;
        case END:
            if (offset > static_cast<sys::Off_T>(mData.size()))
            {
                mPosition = 0;
            }
            else
            {
                mPosition = mData.size() - offset;
            }
            break;
        default:
            mPosition += offset;
            break;
        }

        if (mPosition >= static_cast<sys::Off_T>(mData.size()))
            mPosition = io::InputStream::IS_END;
        return tell();
    }

    /*!
     *  Returns the available bytes to read from the stream
     *  \return the available bytes to read
     */
    sys::Off_T available();

    using OutputStream::write;
    using InputStream::streamTo;

    /*!
     *  Writes the bytes in data to the stream.
     *  \param b the data to write to the stream
     *  \param size the number of bytes to write to the stream
     */
    void write(const sys::byte *b, sys::Size_T size);

    /*!
     * Read up to len bytes of data from this buffer into an array
     * update the mark
     * \param b   Buffer to read into
     * \param len The length to read
     * \throw IoException
     * \return  The number of bytes read
     */
    virtual sys::SSize_T read(sys::byte *b, sys::Size_T len);

    void reset()
    {
        mPosition  = 0;
        mData.resize(0);
    }
    
    /*!
     * Resize the internal buffer
     * \param len the new buffer length
     */
    void
    resize(sys::Size_T len)
    {
        mData.resize(len);
        if (mPosition >= static_cast<sys::Off_T>(len))
            mPosition = len - 1;
    }

    /*!
     * Get a point to the internal buffer 
     * \return pointer to the internal buffer
     */
    sys::ubyte *
    get()
    {
        return mData.empty() ? NULL : &mData[0];
    }

private:
    std::vector<sys::ubyte> mData;
    sys::Off_T mPosition;
};
}

#endif
