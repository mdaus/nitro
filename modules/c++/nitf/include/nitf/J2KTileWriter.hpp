/* =========================================================================
 * This file is part of NITRO
 * =========================================================================
 *
 * (C) Copyright 2017, MDA Information Systems LLC
 * (C) Copyright 2022, Maxar Technologies, Inc.
 *
 * NITRO is free software; you can redistribute it and/or modify
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
 * License along with this program; if not, If not,
 * see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef NITF_J2KTileWriter_hpp_INCLUDED_
#define NITF_J2KTileWriter_hpp_INCLUDED_
#pragma once

#include <io/SeekableStreams.h>
#include <sys/Conf.h>
#include <types/RowCol.h>

#include "nitf/J2KStream.hpp"
#include "nitf/J2KImage.hpp"
#include "nitf/J2KEncoder.hpp"
#include "nitf/J2KCompressionParameters.hpp"

namespace j2k
{
    namespace details
    {
        /*!
         * \class OPJTileWriter
         * \desc Implementation class for writing compressed tiles to an output stream.
         * This class is used by OPJCompressor to do thread based tile compression.
         */
        class TileWriter final
        {
            std::shared_ptr< ::io::SeekableOutputStream> mOutputStream;
            std::shared_ptr<const CompressionParameters> mCompressionParams;

            Stream mStream;             //! The openjpeg stream.
            Image mImage;             //! The openjpeg image.
            Encoder mEncoder;             //! The openjpeg encoder.
            bool mIsCompressing;             //! Whether we are currently compressing or not.

            void resizeTile(types::RowCol<size_t>& tile, size_t tileIndex);

        public:
            /*!
             * Constructor
             *
             * \param outputStream The output stream to write the J2K codestream to.
             *
             * \param compressionParams The J2K compression parameters.
             */
            TileWriter(
                std::shared_ptr< ::io::SeekableOutputStream> outputStream,
                std::shared_ptr<const CompressionParameters> compressionParams);
            TileWriter(const TileWriter&) = delete;
            TileWriter& operator=(const TileWriter&) = delete;
            TileWriter(TileWriter&&) = default;
            TileWriter& operator=(TileWriter&&) = default;

            /*!
             * Destructor - calls end().
             */
            ~TileWriter();

            /*!
             * Starts the J2K compression. The first call to flush() after this
             * is invoked will write the J2K header to the output stream.
             */
            void start();

            /*!
             * Ends the J2K compression. This will flush the J2K footer to the
             * output stream.
             */
            void end();

            /*!
             * Writes any J2K codestream data currently in the internal buffer used by
             * openjpeg to the output stream.
             */
            void flush();

            /*!
             *  Calls opj_write_tile. Tiles should be structured as contiguous,
             *  increasing rows with a fixed column width i.e given 2x6 image
             *  with 2x3 tiles:
             *
             *            0  1  2  3  4  5
             *            6  7  8  9  10 11
             *
             *  The 0th tile:
             *
             *            0  1  2
             *            6  7  8
             *
             *  should be laid out in contiguous memory as: 0 1 2 6 7 8.
             */
            void writeTile(const sys::ubyte* tileData,
                size_t tileIndex);

            /*!
             *  \return true if compression has started, false otherwise.
             */
            bool isCompressing() const
            {
                return mIsCompressing;
            }

            /*!
             * Updates the output stream that openjpeg will write J2K codestream
             * data to when flush() is called.
             *
             * \param outputStream The stream to write to.
             */
            void setOutputStream(
                std::shared_ptr< ::io::SeekableOutputStream> outputStream);
        };
    }
}

#endif NITF_J2KTileWriter_hpp_INCLUDED_