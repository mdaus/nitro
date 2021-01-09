/* =========================================================================
 * This file is part of NITRO
 * =========================================================================
 *
 * (C) Copyright 2004 - 2014, MDA Information Systems LLC
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

#ifndef __NITF_SUBWINDOW_HPP__
#define __NITF_SUBWINDOW_HPP__
#pragma once

#include <vector>
#include <string>

#include "nitf/SubWindow.h"
#include "nitf/DownSampler.hpp"
#include "nitf/Object.hpp"

/*!
 *  \file SubWindow.hpp
 *  \brief  Contains wrapper implementation for SubWindow
 */
namespace nitf
{

/*!
 *  \class SubWindow
 *  \brief  The C++ wrapper for the nitf_SubWindow
 *
 *  This object specifies a sub-window.  It is used as an
 *  argument to image read and write requests to specify
 *  the data requested.
 *
 *  The bandList field specifies the desired bands. Bands are
 *  numbered starting with zero and ordered as the bands
 *  appear physically in the file.  The user must supply
 *  one buffer for each requested band.  Each buffer must
 *  have enough storage for one band of the requested subimage
 *  size.
 *
 *  The DownSampler determines what downsampling (if any) is
 *  performed on this sub-window request.  There is no default
 *  DownSampler.  If you just want to perform pixel skipping, call
 *  setDownSampler() with a PixelSkip object.
 */
DECLARE_CLASS(SubWindow)
{
public:
    //! Copy constructor
    SubWindow(const SubWindow & x);

    //! Assignment Operator
    SubWindow & operator=(const SubWindow & x);

    //! Set native object
    SubWindow(nitf_SubWindow * x);

    //! Constructor
    SubWindow();

    //! Destructor
    ~SubWindow();

    uint32_t getStartRow() const;
    uint32_t getNumRows() const;
    uint32_t getStartCol() const;
    uint32_t getNumCols() const;
    uint32_t getBandList(int i);
    uint32_t getNumBands() const;

    void setStartRow(uint32_t value);
    void setNumRows(uint32_t value);
    void setStartCol(uint32_t value);
    void setNumCols(uint32_t value);
    void setBandList(uint32_t * value);
    void setNumBands(uint32_t value);

    /*!
     * Reference a DownSampler within the SubWindow
     * The SubWindow does NOT own the DownSampler
     * \param downSampler  The down sampler to reference
     */
    void setDownSampler(nitf::DownSampler* downSampler);
    void setDownSampler(nitf::DownSampler& downSampler)
    {
        setDownSampler(&downSampler);
    }

    /*!
     * Return the DownSampler that is referenced by this SubWindow.
     * If no DownSampler is referenced, a NITFException is thrown.
     */
    nitf::DownSampler* getDownSampler();

private:
    nitf::DownSampler* mDownSampler;
    nitf_Error error;
};

inline void setBands(SubWindow& subWindow, std::vector<uint32_t>& bands)
{
    subWindow.setBandList(bands.data());
    subWindow.setNumBands(static_cast<uint32_t>(bands.size()));
}

}
#endif