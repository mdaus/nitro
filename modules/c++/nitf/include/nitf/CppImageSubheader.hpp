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
#include <nitf/CppField.hpp>
#include <nitf/Extensions.hpp>
#include <nitf/System.hpp>
#include <types/Corners.h>
#include <types/LatLon.h>

#ifndef __NITF_CPPIMAGESUBHEADER_HPP__
#define __NITF_CPPIMAGESUBHEADER_HPP__

namespace nitf
{
/*!
 * \class CppImageSubheader
 * \brief C++ implementation of the ImageSubheader
 * Provides methods for getting and setting fields.
 * See Table A-3 of 2500C
 */
class CppImageSubheader
{
public:
    CppImageSubheader();
    /*!
     * \brief Set the pixel type and band related information
     *
     * In addition to the given parameters, the following fields are set:
     * NBANDS
     * XBANDS
     *
     * \param pixelValueType Pixel value type (PVTYPE)
     * \param numBitsPerPixel Number of bits/pixel (NBPP)
     * \param actualNumBitsPerPixel Actual number of bits/pixel (ABPP)
     * \param justification  Pixel justification (PJUST)
     * \param imageRepresentation  Image representation (IREP)
     * \param imageCategory Image category (ICAT)
     * \param bands Vector of BandInfo objects
     */
    void setPixelInformation(
            const std::string& pixelValueType,
            Uint32 numBitsPerPixel,
            Uint32 actualNumBitsPerPixel,
            const std::string& justification,
            const std::string& imageRepresenation,
            const std::string& imageCategory,
            const std::vector<BandInfo>& bands);

    /*!
     *  This function allows the user to set the corner coordinates from a
     *  set of decimal values.  This function only supports CornersTypes of
     *  NITF_GEO or NITF_DECIMAL.  Others will throw an exception
     *
     *  The corners MUST be oriented to correspond to
     *
     *  corners[0] = (0, 0),
     *  corners[1] = (0, MaxCol),
     *  corners[2] = (MaxRow, MaxCol)
     *  corners[3] = (MaxRow, 0)
     *
     *  following in line with 2500C.
     *
     *  \param type NITF_GEO or NITF_DECIMAL
     *  \param corners LatLon corners in specificied order
     */
    void setCornersFromLatLons(CornersType type,
            const types::LatLonCorners& corners);

    inline std::string getCornerCoordinates() const
    {
        return mCornerCoordinates;
    }

    /*!
     * Get the type of corners.  This will return NITF_CORNERS_UNKNOWN
     * in the event that it is not 'D', or 'G'.
     * \return type of corners
     */
    CornersType getCornersType() const;

    /*!
     *  This function allows the user to extract corner coordinates as a
     *  set of decimal values.  This function only supports CornersTypes of
     *  NITF_GEO or NITF_DECIMAL.  Others will throw an exception.
     *
     *  The output corners will be oriented to correspond to
     *  corners[0] = (0, 0),
     *  corners[1] = (0, MaxCol),
     *  corners[2] = (MaxRow, MaxCol)
     *  corners[3] = (MaxRow, 0)
     *
     *  following in line with 2500C.
     *  \return corner coordinates as a set of decimal values
     */
    types::LatLonCorners getCornersAsLatLons() const;

    /*!
     * Set the image dimensions and blocking info.
     * The user specifies the number of rows and columns in the image, number
     * of rows and columns per block, and blocking mode. The number of blocks
     * per row and column is calculated. The NITF 2500C large block option can
     * be selected for either dimension by setting the corresponding block
     * dimension to 0.
     *
     * \param numRows           The number of rows
     * \param numCols           The number of columns
     * \param numRowsPerBlock   The number of rows/block
     * \param numColsPerBlock   The number of columns/block
     * \param imageMode         Image mode
     */
    void setBlocking(Uint32 numRows,
            Uint32 numCols,
            Uint32 numRowsPerBlock,
            Uint32 numColsPerBlock,
            const std::string& imageMode);

    /*!
     * Set the image dimensions and blocking.
     *
     * The blocking is set to the simplest possible blocking given the
     * dimension, no blocking if possible. Blocking mode is set to B. Whether
     * or not blocking masks are generated is dependent on the compression
     * code at the time the image is written. NITF 2500C large block
     * dimensions are not set by this function. The more general set blocking
     * function must be used to specify this blocking option.
     *
     * \param numRows           The number of rows
     * \param numCols           The number of columns
     */
    void setDimensions(Uint32 numRows, Uint32 numCols);

    //! Get the number of bands
    Uint32 getBandCount() const;

    /*!
     * Creates the specified number of bandInfo objects, and updates the
     * bandInfo array. This also updates the NBANDS/XBANDS fields in the
     * ImageSubheader.
     */
    void createBands(Uint32 bandCount);

    //! Insert the given comment at the given index (zero-indexed);
    void insertImageComment(const std::string& comment, size_t position);

    //! Remove the given comment at the given index (zero-indexed);
    void removeImageComment(size_t position);

    inline Uint32 getNumImageComments() const
    {
        return mNumImageComments;
    }

    inline const std::vector<BCSA<std::string, 80> >& getImageComments() const
    {
        return mImageComments;
    }

    void setImageCompression(const std::string& imageCompression);
    inline std::string getImageCompression() const
    {
        return mImageCompression;
    }

    inline std::string getCompressionRate() const
    {
        return mCompressionRate;
    }

    inline void setCompressionRate(const std::string& compressionRate)
    {
        mCompressionRate = compressionRate;
    }

    inline Uint32 getImageSyncCode() const
    {
        return mImageSyncCode;
    }

    inline std::string getFilePartType() const
    {
        return mFilePartType;
    }

    inline std::string getImageId() const
    {
        return mImageId;
    }

    inline void setImageId()
    {
        setImageId();
    }

    inline DateTime getImageDateAndTime() const
    {
        return mImageDateAndTime;
    }

    inline void setImageDateAndTime(const DateTime& datetime)
    {
        mImageDateAndTime = datetime;
    }

    inline std::string getTargetId() const
    {
        return mTargetId;
    }

    inline void setTargetId(const std::string& targetId)
    {
        mTargetId = targetId;
    }

    inline std::string getImageTitle() const
    {
        return mImageTitle;
    }

    inline void setImageTitle(const std::string& imageTitle)
    {
        mImageTitle = imageTitle;
    }

    inline std::string getImageSecurityClassification() const
    {
        return mImageSecurityClassification;
    }

    void setImageSecurityClassification(const std::string& classification);

    inline std::string getClassificationSystem() const
    {
        return mClassificationSystem;
    }

    inline void setClassificationSystem(const std::string& classificationSystem)
    {
        mClassificationSystem = classificationSystem;
    }

    inline std::string getCodewords() const
    {
        return mCodewords;
    }

    inline void setCodewords(const std::string& codewords)
    {
        mCodewords = codewords;
    }

    inline std::string getControlAndHandling() const
    {
        return mControlAndHandling;
    }

    inline void setControlAndHandling(const std::string& controlAndHandling)
    {
        mControlAndHandling = controlAndHandling;
    }

    inline std::string getReleasingInstructions() const
    {
        return mReleasingInstructions;
    }

    inline void setReleasingInstructions(const std::string& instructions)
    {
        mReleasingInstructions = instructions;
    }

    void setDeclassificationType(const std::string& declassType);
    inline std::string getDeclassificationType() const
    {
        return mDeclassificationType;
    }

    inline DateTime getDeclassificationDate() const
    {
        return mDeclassificationDate;
    }

    inline void setDeclassificationDate(const DateTime& date)
    {
        mDeclassificationDate = date;
    }

    void setDeclassificationExemption(const std::string& exemption);
    inline std::string getDeclassificationExemption() const
    {
        return mDeclassificationExemption;
    }

    void setDowngrade(const std::string& downgrade);
    inline std::string getDowngrade() const
    {
        return mDowngrade;
    }

    inline DateTime getDowngradeDateTime() const
    {
        return mDowngradeDateTime;
    }

    inline void setDowngradeDateTime(const DateTime& datetime)
    {
        mDowngradeDateTime = datetime;
    }

    inline std::string getClassificationText() const
    {
        return mClassificationText;
    }

    inline void setClassificationText(const std::string& text)
    {
        mClassificationText = text;
    }

    void setClassificationAuthorityType(const std::string& classificationType);
    inline std::string getClassificationAuthorityType() const
    {
        return mClassificationAuthorityType;
    }

    inline std::string getClassificationAuthority() const
    {
        return mClassificationAuthority;
    }

    inline void setClassificationAuthority(const std::string& authority)
    {
        mClassificationAuthority = authority;
    }

    void setClassificationReason(const std::string& reason);
    inline std::string getClassificationReason() const
    {
        return mClassificationReason;
    }

    inline DateTime getSecuritySourceDate() const
    {
        return mSourceDate;
    }

    inline void setSecuritySourceDate(const DateTime& sourceDate)
    {
        mSourceDate = sourceDate;
    }

    inline std::string getSecurityControlNumber() const
    {
        return mControlNumber;
    }

    inline void setSecurityControlNumber(const std::string& controlNumber)
    {
        mControlNumber = controlNumber;
    }

    // 2500C says the value is 0, so no reason to set anything
    inline Uint32 getEncrypted() const
    {
        return mEncryption;
    }

    inline std::string getImageSource() const
    {
        return mImageSource;
    }

    inline void setImageSource(const std::string& imageSource)
    {
        mImageSource = imageSource;
    }

    inline std::string getImageMode() const
    {
        return mImageMode;
    }

    inline Uint32 getNumRows() const
    {
        return mNumRows;
    }

    inline Uint32 getNumCols() const
    {
        return mNumCols;
    }

    inline Uint32 getNumPixelsPerHorizBlock() const
    {
        return mNumPixelsPerHorizBlock;
    }

    inline Uint32 getNumPixelsPerVertBlock() const
    {
        return mNumPixelsPerVertBlock;
    }

    inline Uint32 getNumBlocksPerRow() const
    {
        return mNumBlocksPerRow;
    }

    inline Uint32 getNumBlocksPerCol() const
    {
        return mNumBlocksPerCol;
    }

    inline std::string getImageCoordinateSystem() const
    {
        return mImageCoordinateSystem;
    }

    inline std::string getPixelValueType() const
    {
        return mPixelValueType;
    }

    inline std::string getPixelJustification() const
    {
        return mPixelJustification;
    }

    inline Uint32 getNumBitsPerPixel() const
    {
        return mNumBitsPerPixel;
    }

    inline Uint32 getActualBitsPerPixel() const
    {
        return mActualBitsPerPixel;
    }

    inline Uint32 getNumImageBands() const
    {
        return mNumImageBands;
    }

    inline Uint32 getNumMultispectralImageBands() const
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

    inline BandInfo getBandInfo(Uint32 bandNumber) const
    {
        return mBandInfos[bandNumber];
    }

    inline Uint32 getImageDisplayLevel() const
    {
        return mImageDisplayLevel;
    }

    inline void setImageDisplayLevel(Uint32 imageDisplayLevel)
    {
        mImageDisplayLevel = imageDisplayLevel;
    }

    inline Uint32 getAttachmentLevel() const
    {
        return mAttachmentLevel;
    }

    inline void setAttachmentLevel(Uint32 attachmentLevel)
    {
        mAttachmentLevel = attachmentLevel;
    }

    void setImageLocation(const std::string& imageLocation);
    void setImageLocation(Int32 row, Int32 col);
    inline std::string getImageLocation() const
    {
        return mImageLocation;
    }

    void setImageMagnification(const std::string& imageMagnification);
    inline std::string getImageMagnification() const
    {
        return mImageMagnification;
    }

    inline Uint32 getUserDefinedImageDataLength() const
    {
        return mUserDefinedImageDataLength;
    }

    // TODO: Should this be private and set when w/ user-defined section?
    inline void setUserDefinedImageData(Uint32 size)
    {
        mUserDefinedImageDataLength = size;
    }

    inline Uint32 getUserDefinedOverflow() const
    {
        return mUserDefinedOverflow;
    }

    inline void setUserDefinedOverflow(Uint32 overflow)
    {
        mUserDefinedOverflow = overflow;
    }

    inline Extensions getUserDefinedImageData() const
    {
        return mUserDefinedImageData;
    }

    inline void setUserDefinedImageData(const Extensions& userDefinedData)
    {
        mUserDefinedImageData = userDefinedData;
    }

    inline Uint32 getImageExtendedSubheaderDataLength() const
    {
        return mImageExtendedSubheaderDataLength;
    }

    // TODO: Should this be private and set when w/ user-defined section?
    inline void setImageExtendedSubheaderDataLength(Uint32 size)
    {
        mImageExtendedSubheaderDataLength = size;
    }

    inline Uint32 getImageExtendedSubheaderOverflow() const
    {
        return mImageExtendedSubheaderOverflow;
    }

    inline void setImageExtendedSubheaderOverflow(Uint32 size)
    {
        mImageExtendedSubheaderOverflow = size;
    }

    inline Extensions getImageExtendedSubheaderData() const
    {
        return mImageExtendedSubheaderData;
    }

    inline void setImageExtendedSubheaderData(const Extensions& extendedData)
    {
        mImageExtendedSubheaderData = extendedData;
    }


private:
    template<typename T>
    void ensureRange(const std::string& fieldName,
            T lowerBound, T upperBound, T value)
    {
        if (value < lowerBound || value > upperBound)
        {
            throw except::Exception(Ctxt(fieldName + " must be in range [" +
                        str::toString(lowerBound) + ":" +
                        str::toString(upperBound) + "]. Got " +
                        str::toString(value)));
        }
    }
    Uint32 computeBlockCount(
            Uint32 numElements,
            Uint32* elementsPerBlock) const;
    static bool isLocationCoordinateValid(const std::string& coordinate);
    void setImageMode(const std::string& imageMode);
    void setNumBands(Uint32 numBands);
    void setNumRows(Uint32 numRows);
    void setNumCols(Uint32 numCols);
    void setNumPixelsPerHorizBlock(Uint32 numPixels);
    void setNumPixelsPerVertBlock(Uint32 numPixels);
    void setNumBlocksPerRow(Uint32 numBlocksPerRow);
    void setNumBlocksPerCol(Uint32 numBlocksPerCol);
    void setPixelValueType(const std::string& pixelValueType);
    void setPixelJustification(const std::string& justification);
    void setNumBitsPerPixel(Uint32 bitsPerPixel);
    void setActualBitsPerPixel(Uint32 actualBitsPerPixel);
    void setImageCategory(const std::string& category);
    void setImageRepresentation(const std::string& representation);
    void setNumImageBands(Uint32 numImageBands);
    void setNumMultispectralImageBands(Uint32 numBands);

private:
    const BCSA<std::string, 2> mFilePartType;
    BCSA<std::string, 10> mImageId;
    BCSA<DateTime, 14> mImageDateAndTime;
    BCSA<std::string, 17> mTargetId;
    BCSA<std::string, 80> mImageTitle;
    BCSA<std::string, 1> mImageSecurityClassification;
    // TODO: mImageClassificationSystem -> mControlNumber
    // might need their own class
    BCSA<std::string, 2> mClassificationSystem;
    BCSA<std::string, 11> mCodewords;
    BCSA<std::string, 2> mControlAndHandling;
    BCSA<std::string, 20> mReleasingInstructions;
    BCSA<std::string, 2> mDeclassificationType;
    BCSA<DateTime, 8> mDeclassificationDate;
    BCSA<std::string, 4> mDeclassificationExemption;
    BCSA<std::string, 1> mDowngrade;
    BCSA<DateTime, 8> mDowngradeDateTime;
    BCSA<std::string, 43> mClassificationText;
    BCSA<std::string, 1> mClassificationAuthorityType;
    BCSA<std::string, 40> mClassificationAuthority;
    BCSA<std::string, 1> mClassificationReason;
    BCSA<DateTime, 8> mSourceDate;
    BCSA<std::string, 15> mControlNumber;

    BCSN<Uint32, 1> mEncryption;
    BCSA<std::string, 42> mImageSource;
    BCSN<Uint32, 8> mNumRows;
    BCSN<Uint32, 8> mNumCols;
    BCSA<std::string, 3> mPixelValueType;
    BCSA<std::string, 8> mImageRepresentation;
    BCSA<std::string, 8> mImageCategory;
    BCSN<Uint32, 2> mActualBitsPerPixel;
    BCSA<std::string, 1> mPixelJustification;
    BCSA<std::string, 1> mImageCoordinateSystem;
    BCSA<std::string, 60> mCornerCoordinates;
    BCSN<Uint32, 1> mNumImageComments;
    std::vector<BCSA<std::string, 80> > mImageComments;
    BCSA<std::string, 2> mImageCompression;
    BCSA<std::string, 4> mCompressionRate;
    BCSN<Uint32, 1> mNumImageBands;
    BCSN<Uint32, 5> mNumMultispectralImageBands;
    std::vector<BandInfo> mBandInfos;
    BCSN<Uint32, 1> mImageSyncCode;
    BCSA<std::string, 1> mImageMode;
    BCSN<Uint32, 4> mNumBlocksPerRow;
    BCSN<Uint32, 4> mNumBlocksPerCol;
    BCSN<Uint32, 4> mNumPixelsPerHorizBlock;
    BCSN<Uint32, 4> mNumPixelsPerVertBlock;
    BCSN<Uint32, 2> mNumBitsPerPixel;
    BCSN<Uint32, 3> mImageDisplayLevel;
    BCSN<Uint32, 3> mAttachmentLevel;
    BCSN<std::string, 10> mImageLocation;
    BCSA<std::string, 4> mImageMagnification;
    BCSN<Uint32, 5> mUserDefinedImageDataLength;
    BCSN<Uint32, 3> mUserDefinedOverflow;
    Extensions mUserDefinedImageData;
    BCSN<Uint32, 5> mImageExtendedSubheaderDataLength;
    BCSN<Uint32, 3> mImageExtendedSubheaderOverflow;
    Extensions mImageExtendedSubheaderData;

    static const size_t NITF_BLOCK_DIM_MAX = 8192;
    static const size_t NITF_BLOCK_DEFAULT_MAX = 1024;
    static const size_t NITF_BLOCK_DEFAULT_MIN = 1024;
};
}

#endif

