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

#include <nitf/ImageSubheader.h>
#include <nitf/CppImageSubheader.hpp>
#include <nitf/Utils.hpp>

namespace nitf
{
CppImageSubheader::CppImageSubheader() :
    mFilePartType("IM"),
    mEncryption(0),
    mImageCategory("VIS"),
    mPixelJustification("R"),
    mImageSyncCode(0),
    mImageMagnification("1.0 ")
{
}

void CppImageSubheader::setPixelInformation(
        const std::string& pixelValueType,
        Uint32 numBitsPerPixel,
        Uint32 actualNumBitsPerPixel,
        const std::string& justification,
        const std::string& imageRepresenation,
        const std::string& imageCategory,
        const std::vector<BandInfo>& bands)
{
    setPixelValueType(pixelValueType);
    setPixelJustification(justification);
    setNumBitsPerPixel(numBitsPerPixel);
    setActualBitsPerPixel(actualNumBitsPerPixel);
    setImageRepresentation(imageRepresenation);
    setImageCategory(imageCategory);
    setNumBands(bands.size());
    mBandInfos.clear();
    mBandInfos.resize(bands.size());
    for (size_t ii = 0; ii < bands.size(); ++ii)
    {
        mBandInfos[ii] = bands[ii];
    }
}

Uint32 CppImageSubheader::computeBlockCount(
        Uint32 numElements,
        Uint32* elementsPerBlock) const
{
    Uint32 blockCount = 1;
    if (*elementsPerBlock > NITF_BLOCK_DIM_MAX)
    {
        *elementsPerBlock = 0;
    }

    if (*elementsPerBlock != 0)
    {
        blockCount = numElements / *elementsPerBlock;
        if (numElements % *elementsPerBlock != 0)
        {
            // Round up for any left over
            ++blockCount;
        }
    }
    return blockCount;
}

void CppImageSubheader::setBlocking(Uint32 numRows,
        Uint32 numCols,
        Uint32 numRowsPerBlock,
        Uint32 numColsPerBlock,
        const std::string& imageMode)
{
    setNumRows(numRows);
    setNumCols(numCols);
    const Uint32 numBlocksPerCol = computeBlockCount(numRows,
            &numRowsPerBlock);
    const Uint32 numBlocksPerRow = computeBlockCount(numCols,
            &numColsPerBlock);
    setNumPixelsPerHorizBlock(numColsPerBlock);
    setNumPixelsPerVertBlock(numRowsPerBlock);
    setNumBlocksPerCol(numBlocksPerCol);
    setNumBlocksPerRow(numBlocksPerRow);
    setImageMode(imageMode);
}

void CppImageSubheader::setDimensions(Uint32 numRows,
        Uint32 numCols)
{
    // TODO: Nitro calls selectBlockSize here, which tries to find
    // the block size with the least padding
    // Because of how the defaults are, it never actually does anything,
    // so I'm assuming it's not important
    const size_t numRowsPerBlock =
        (numRows <= NITF_BLOCK_DIM_MAX) ? numRows :
        NITF_BLOCK_DEFAULT_MAX;
    const size_t numColsPerBlock =
        (numCols <= NITF_BLOCK_DIM_MAX) ? numCols :
        NITF_BLOCK_DEFAULT_MAX;
    setBlocking(numRows, numCols, numRowsPerBlock, numColsPerBlock, "B");
}

void CppImageSubheader::createBands(Uint32 bandCount)
{
    const size_t currentCount = getBandCount();
    const size_t totalBandCount = currentCount + bandCount;
    if (totalBandCount > NITF_MAX_BAND_COUNT)
    {
        throw except::Exception(Ctxt("Tried to create too many bands"));
    }
    mBandInfos.resize(totalBandCount);
    setNumBands(totalBandCount);
}

void CppImageSubheader::setNumBands(Uint32 numBands)
{
    if (numBands <= 9)
    {
        setNumImageBands(numBands);
        setNumMultispectralImageBands(0);
    }
    else
    {
        setNumImageBands(0);
        setNumMultispectralImageBands(numBands);
    }
}

void CppImageSubheader::insertImageComment(const std::string& comment,
        size_t position)
{
    if (mImageComments.size() >= 9)
    {
        throw except::Exception(Ctxt("Cannot insert any more comments"));
    }
    if (position > mNumImageComments)
    {
        position = mNumImageComments;
    }
    mImageComments.insert(mImageComments.begin() + position, comment);
    mNumImageComments = mNumImageComments + 1;
}

void CppImageSubheader::removeImageComment(size_t position)
{
    if (position >= mNumImageComments)
    {
        throw except::Exception(Ctxt("No comment at position " + position));
    }
    mImageComments.erase(mImageComments.begin() + position);
    mNumImageComments = mNumImageComments - 1;
}

Uint32 CppImageSubheader::getBandCount() const
{
    return mNumImageBands ? mNumImageBands : mNumMultispectralImageBands;
}

void CppImageSubheader::setImageMode(const std::string& imageMode)
{
    if (imageMode != "B" &&
        imageMode != "P" &&
        imageMode != "R" &&
        imageMode != "S")
    {
        throw except::Exception(Ctxt("Invalid imageMode: " + imageMode));
    }

    mImageMode = imageMode;
}

void CppImageSubheader::setImageSecurityClassification(
        const std::string& classification)
{
    if (classification != "T" &&
        classification != "S" &&
        classification != "C" &&
        classification != "R" &&
        classification != "U")
    {
        throw except::Exception(Ctxt(
                "Invalid classification: " + classification));
    }
    mImageSecurityClassification = classification;
}

void CppImageSubheader::setDeclassificationType(const std::string& declassType)
{
    if (declassType != "DD" &&
        declassType != "DE" &&
        declassType != "GD" &&
        declassType != "GE" &&
        declassType != "O" &&
        declassType != "X" &&
        declassType != "  " &&
        !declassType.empty())
    {
        throw except::Exception(Ctxt(
                    "Invalid declassificationType: " + declassType));
    }
    mDeclassificationType = declassType;
}

void CppImageSubheader::setDeclassificationExemption(
        const std::string& exemption)
{
    if (exemption != "X1" && exemption != "X2" &&
        exemption != "X3" && exemption != "X4" &&
        exemption != "X5" && exemption != "X6" &&
        exemption != "X7" && exemption != "X8" &&
        exemption != "X251" && exemption != "X252" &&
        exemption != "X253" && exemption != "X254" &&
        exemption != "X255" && exemption != "X256" &&
        exemption != "X257" && exemption != "X258" &&
        exemption != "X259" && exemption != "    " &&
        !exemption.empty())
    {
        throw except::Exception(Ctxt(
                "Invalid declassificationExemption: " + exemption));
    }
    mDeclassificationExemption = exemption;
}

void CppImageSubheader::setDowngrade(const std::string& downgrade)
{
    if (downgrade != "S" && downgrade != "C" &&
        downgrade != "R" && downgrade != " " &&
        !downgrade.empty())
    {
        throw except::Exception(Ctxt(
                    "Invalid downgrade: " + downgrade));
    }
    mDowngrade = downgrade;
}

void CppImageSubheader::setClassificationAuthorityType(
        const std::string& classificationType)
{
    if (classificationType != "O" &&
        classificationType != "D" &&
        classificationType != "M" &&
        classificationType != " " &&
        !classificationType.empty())
    {
        throw except::Exception(Ctxt(
                "Invalid classificationAuthorityType: " + classificationType));
    }
    mClassificationAuthorityType = classificationType;
}

void CppImageSubheader::setClassificationReason(const std::string& reason)
{
    const char initial = reason[0];
    if ((initial < 'A' || initial > 'G') &&
        initial != ' ')
    {
        throw except::Exception(Ctxt(
                "Invalid classificationReason: " + reason));
    }
    mClassificationReason = reason;
}

void CppImageSubheader::setCornersFromLatLons(CornersType type,
        const types::LatLonCorners& corners)
{
    std::string cornersString;
    switch(type)
    {
        case NITF_CORNERS_DECIMAL:
            for (size_t ii = 0; ii < 4; ++ii)
            {
                const types::LatLon& corner = corners.getCorner(ii);
                cornersString += Utils::decimalLatToString(corner.getLat());
                cornersString += Utils::decimalLonToString(corner.getLon());
            }
            break;
        case NITF_CORNERS_GEO:
            for (size_t ii = 0; ii < 4; ++ii)
            {
                const types::LatLon& corner = corners.getCorner(ii);
                cornersString += Utils::decimalLatToGeoString(corner.getLat());
                cornersString += Utils::decimalLonToGeoString(corner.getLon());
            }
            break;
        default:
            throw except::Exception(Ctxt("Can only support IGEOLO 'D' or 'G' "
                        "for this operation"));
    }
    mCornerCoordinates = cornersString;
    mImageCoordinateSystem = std::string(1, Utils::cornersTypeAsCoordRep(type));
}

types::LatLonCorners CppImageSubheader::getCornersAsLatLons() const
{
    types::LatLonCorners corners;
    const std::string coordinates = getCornerCoordinates();
    switch(getCornersType())
    {
    case NITF_CORNERS_GEO:
        for (size_t ii = 0; ii < 4; ++ii)
        {
            const std::pair<std::string, std::string> coordinatePair =
                    Utils::splitIgeolo(coordinates, ii);
            corners.getCorner(ii).setLat(Utils::geographicStringToDecimal(
                        coordinatePair.first));
            corners.getCorner(ii).setLon(Utils::geographicStringToDecimal(
                        coordinatePair.second));
        }
        break;
    case NITF_CORNERS_DECIMAL:
        for (size_t ii = 0; ii < 4; ++ii)
        {
            const std::pair<std::string, std::string> coordinatePair =
                    Utils::splitIgeolo(coordinates, ii);
            corners.getCorner(ii).setLat(Utils::parseDecimalString(
                        coordinatePair.first));
            corners.getCorner(ii).setLon(Utils::parseDecimalString(
                        coordinatePair.second));
        }
        break;
    default:
        throw except::Exception(Ctxt("Can only support IGEOLO 'D' or 'G' for "
                    "this operation. Found " + getImageCoordinateSystem()));
    }
    return corners;
}

CornersType CppImageSubheader::getCornersType() const
{
    CornersType type = NITF_CORNERS_UNKNOWN;
    // TODO: NITRO has other corner types
    // But could they ever actually happen?
    switch (mImageCoordinateSystem.toString()[0])
    {
        case 'D':
            type = NITF_CORNERS_DECIMAL;
            break;
        case 'G':
            type = NITF_CORNERS_GEO;
            break;
    }
    return type;
}

void CppImageSubheader::setPixelValueType(const std::string& pixelValueType)
{
    if (pixelValueType != "INT" &&
        pixelValueType != "B" &&
        pixelValueType != "SI" &&
        pixelValueType != "R" &&
        pixelValueType != "C")
    {
        throw except::Exception(Ctxt(
                "Invalid pixelValueType: " + pixelValueType));
    }
    mPixelValueType = pixelValueType;
}

void CppImageSubheader::setPixelJustification(const std::string& justification)
{
    if (justification != "L" && justification != "R")
    {
        throw except::Exception(Ctxt(
                "Invalid pixelJustification: " + justification));
    }
    mPixelJustification = justification;
}

void CppImageSubheader::setImageRepresentation(
        const std::string& representation)
{
    if (representation != "MONO" &&
        representation != "RGB" &&
        representation != "RGB/LUT" &&
        representation != "MULTI" &&
        representation != "NODISPLY" &&
        representation != "NVECTOR" &&
        representation != "POLAR" &&
        representation != "VPH" &&
        representation != "YCbCr601")
    {
        throw except::Exception(Ctxt(
                "Invalid image representation: " + representation));
    }
    mImageRepresentation = representation;
}

void CppImageSubheader::setImageCategory(const std::string& category)
{
    if (category != "VIS" &&
        category != "SL" &&
        category != "TI" &&
        category != "FL" &&
        category != "RD" &&
        category != "EO" &&
        category != "OP" &&
        category != "HR" &&
        category != "HS" &&
        category != "CP" &&
        category != "BP" &&
        category != "SAR" &&
        category != "SARIQ" &&
        category != "IR" &&
        category != "MAP" &&
        category != "MS" &&
        category != "FP" &&
        category != "MRI" &&
        category != "XRAY" &&
        category != "CAT" &&
        category != "VD" &&
        category != "PAT" &&
        category != "LEG" &&
        category != "DTEM" &&
        category != "MATR" &&
        category != "LOCG" &&
        category != "BARO" &&
        category != "CURRENT" &&
        category != "DEPTH" &&
        category != "WIND")
    {
        throw except::Exception(Ctxt("Invalid category: " + category));
    }
    mImageCategory = category;
}

void CppImageSubheader::setImageCompression(const std::string& imageCompression)
{
    if (imageCompression != "NC" && imageCompression != "NM" &&
        imageCompression != "C1" && imageCompression != "C3" &&
        imageCompression != "C4" && imageCompression != "C5" &&
        imageCompression != "C6" && imageCompression != "C7" &&
        imageCompression != "C8" && imageCompression != "I1" &&
        imageCompression != "M1" && imageCompression != "M3" &&
        imageCompression != "M4" && imageCompression != "M5" &&
        imageCompression != "M6" && imageCompression != "M7" &&
        imageCompression != "M8")
    {
        throw except::Exception(Ctxt(
                "Invalid imageCompression: " + imageCompression));
    }
    mImageCompression = imageCompression;
}

void CppImageSubheader::setNumBitsPerPixel(Uint32 bitsPerPixel)
{
    ensureRange<Uint32>("NBPP", 1, 96, bitsPerPixel);
    mNumBitsPerPixel = bitsPerPixel;
}

void CppImageSubheader::setActualBitsPerPixel(Uint32 actualBitsPerPixel)
{
    ensureRange<Uint32>("ABPP", 1, 96, actualBitsPerPixel);
    mActualBitsPerPixel = actualBitsPerPixel;
}

void CppImageSubheader::setNumRows(Uint32 numRows)
{
    ensureRange<Uint32>("NROWS", 1, 99999999, numRows);
    mNumRows = numRows;
}

void CppImageSubheader::setNumCols(Uint32 numCols)
{
    ensureRange<Uint32>("NCOLS", 1, 99999999, numCols);
    mNumCols = numCols;
}

void CppImageSubheader::setNumPixelsPerHorizBlock(Uint32 numPixels)
{
    ensureRange<Uint32>("NPPHB", 0, NITF_BLOCK_DIM_MAX, numPixels);
    mNumPixelsPerHorizBlock = numPixels;
}

void CppImageSubheader::setNumPixelsPerVertBlock(Uint32 numPixels)
{
    ensureRange<Uint32>("NPPVB", 0, NITF_BLOCK_DIM_MAX, numPixels);
    mNumPixelsPerVertBlock = numPixels;
}

void CppImageSubheader::setNumBlocksPerRow(Uint32 numBlocksPerRow)
{
    ensureRange<Uint32>("NBPR", 1, 9999, numBlocksPerRow);
    mNumBlocksPerRow = numBlocksPerRow;
}

void CppImageSubheader::setNumBlocksPerCol(Uint32 numBlocksPerCol)
{
    ensureRange<Uint32>("NBPC", 1, 9999, numBlocksPerCol);
    mNumBlocksPerCol = numBlocksPerCol;
}

void CppImageSubheader::setNumImageBands(Uint32 numImageBands)
{
    ensureRange<Uint32>("NBANDS", 0, 9, numImageBands);
    mNumImageBands = numImageBands;
}

void CppImageSubheader::setNumMultispectralImageBands(Uint32 numBands)
{
    if (numBands != 0)
    {
        ensureRange<Uint32>("XBANDS", 10, 99999, numBands);
    }
    mNumMultispectralImageBands = numBands;
}

bool CppImageSubheader::isLocationCoordinateValid(const std::string& coordinate)
{
    if (str::isNumeric(coordinate))
    {
        return true;
    }
    if (str::startsWith(coordinate, "-"))
    {
        return str::isNumeric(coordinate.substr(1, 4));
    }
    return false;
}

void CppImageSubheader::setImageLocation(const std::string& imageLocation)
{
    if (imageLocation.size() != 10 ||
        !isLocationCoordinateValid(imageLocation.substr(0, 5)) ||
        !isLocationCoordinateValid(imageLocation.substr(5, 5)))
    {
        throw except::Exception(Ctxt(
                "Invalid image location: " + imageLocation));
    }
    mImageLocation = imageLocation;
}

void CppImageSubheader::setImageLocation(Int32 row, Int32 col)
{
    BCSN<Int32, 5> rowField(row);
    BCSN<Int32, 5> colField(col);
    mImageLocation = rowField.toString() + colField.toString();
}

void CppImageSubheader::setImageMagnification(
        const std::string& imageMagnification)
{
    bool isValid = true;
    if (imageMagnification == "1.0 " || imageMagnification == "1.0")
    {
        // Valid default value; don't need to do anything
    }
    else if (!str::startsWith(imageMagnification, "/"))
    {
        isValid = false;
    }
    else
    {
        isValid = str::isNumeric(imageMagnification.substr(1,
                imageMagnification.size() - 1));
    }

    if (!isValid)
    {
        throw except::Exception(Ctxt(
                "Invalid magnification: " + imageMagnification));
    }
    mImageMagnification = imageMagnification;
}
}

