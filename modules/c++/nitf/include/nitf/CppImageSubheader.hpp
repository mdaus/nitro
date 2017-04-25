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


#include <string>
#include <vector>
#include <nitf/BandInfo.hpp>
#include <nitf/System.hpp>

#ifndef __NITF_CPPIMAGESUBHEADER_HPP__
#define __NITF_CPPIMAGESUBHEADER_HPP__

namespace nitf
{
class CppImageSubheader
{
public:
    void setPixelInformation(const std::string& pixelValueType, //TODO: Enum?
            nitf::Uint32 numBitsPerPixel,
            nitf::Uint32 actualNumBitsPerPixel,
            const std::string& justification,
            const std::string& imageRepresenation,
            const std::string& imageCategory,
            const std::vector<nitf::BandInfo>& bands);

    inline std::string getPixelValueType() const
    {
        return mPixelValueType;
    }

    inline std::string getPixelJustification() const
    {
        return mPixelJustification;
    }

    inline nitf::Uint32 getNumBitsPerPixel() const
    {
        return mNumBitsPerPixel;
    }

    inline nitf::Uint32 getActualBitsPerPixel() const
    {
        return mActualBitsPerPixel;
    }

    inline nitf::Uint32 getNumImageBands() const
    {
        //TODO: Is there a better way to hold this?
        return mNumImageBands;
    }

    inline nitf::Uint32 getNumMultispectralImageBands() const
    {
        return mNumMultispectralImageBands;
    }

    inline std::string getImageRepresentation() const
    {
        return mImageRepresentation;
    }

    inline std::string getImageCategory() const
    {
        return mImageCategory;
    }

    inline nitf::BandInfo getBandInfo(nitf::Uint32 bandNumber) const
    {
        return mBandInfos[bandNumber];
    }

private:
    std::string mPixelValueType;
    std::string mPixelJustification;
    std::string mImageRepresentation;
    std::string mImageCategory;
    nitf::Uint32 mNumBitsPerPixel;
    nitf::Uint32 mActualBitsPerPixel;
    nitf::Uint32 mNumImageBands;
    nitf::Uint32 mNumMultispectralImageBands;
    std::vector<nitf::BandInfo> mBandInfos;



};
}

#endif

