/* =========================================================================
 * This file is part of NITRO
 * =========================================================================
 *
 * (C) Copyright 2004 - 2017, MDA Information Systems LLC
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

#include <nitf/CppImageSubheader.hpp>

namespace nitf
{
    void CppImageSubheader::setPixelInformation(
            const std::string& pixelValueType, //TODO: Enum?
            nitf::Uint32 numBitsPerPixel,
            nitf::Uint32 actualNumBitsPerPixel,
            const std::string& justification,
            const std::string& imageRepresenation,
            const std::string& imageCategory,
            const std::vector<nitf::BandInfo>& bands)
    {
        mPixelValueType = pixelValueType;
        mPixelJustification = justification;
        mNumBitsPerPixel = numBitsPerPixel;
        mActualBitsPerPixel = actualNumBitsPerPixel;
        mNumMultispectralImageBands = 0;
        mImageRepresentation = imageRepresenation;
        mImageCategory = imageCategory;

        if (bands.size() > 9)
        {
            mNumImageBands = 0;
            mNumMultispectralImageBands = bands.size();
        }
        else
        {
            mNumImageBands = bands.size();
            mNumMultispectralImageBands = 0;
        }

        mBandInfos.clear();
        mBandInfos.resize(bands.size());
        for (size_t ii = 0; ii < bands.size(); ++ii)
        {
            mBandInfos[ii] = bands[ii];
        }
    }

}

