/* =========================================================================
 * This file is part of types-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2017, MDA Information Systems LLC
 *
 * types-c++ is free software; you can redistribute it and/or modify
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

#ifndef __TYPES_LAT_LON_H__
#define __TYPES_LAT_LON_H__

#include <math/Constants.h>

namespace types
{
class LatLon
{
public:
    LatLon(double scalar = 0.0) :
        mLat(scalar), mLon(scalar)
    {
    }

    LatLon(double lat, double lon)
        : mLat(lat), mLon(lon)
    {
    }

    LatLon(const LatLon& lla)
    {
        mLat = lla.mLat;
        mLon = lla.mLon;
    }

    LatLon& operator=(const LatLon& lla)
    {
        if (this != &lla)
        {
            mLat = lla.mLat;
            mLon = lla.mLon;
        }
        return *this;
    }

    double getLat() const
    {
        return mLat;
    }
    double getLon() const
    {
        return mLon;
    }

    double getLatRadians() const
    {
        return mLat * math::Constants::DEGREES_TO_RADIANS;
    }
    double getLonRadians() const
    {
        return mLon * math::Constants::DEGREES_TO_RADIANS;
    }

    void setLat(double lat)
    {
        mLat = lat;
    }
    void setLon(double lon)
    {
        mLon = lon;
    }

    void setLatRadians(double lat)
    {
        mLat = (lat * math::Constants::RADIANS_TO_DEGREES);
    }

    void setLonRadians(double lon)
    {
        mLon = (lon * math::Constants::RADIANS_TO_DEGREES);
    }

    bool operator==(const LatLon& x) const
    {
        return mLat == x.mLat && mLon == x.mLon;
    }

    bool operator!=(const LatLon& x) const
    {
        return !(*this == x);
    }

protected:
    double mLat;
    double mLon;
};

class LatLonAlt : public LatLon
{
public:
    LatLonAlt(double scalar = 0.0) : LatLon(scalar), mAlt(scalar)
    {
    }

    LatLonAlt(double lat, double lon, double alt = 0)
        : LatLon(lat, lon), mAlt(alt)
    {
    }

    LatLonAlt(const LatLonAlt& lla)
    {
        mLat = lla.mLat;
        mLon = lla.mLon;
        mAlt = lla.mAlt;
    }

    LatLonAlt& operator=(const LatLonAlt& lla)
    {
        if (this != &lla)
        {
            mLat = lla.mLat;
            mLon = lla.mLon;
            mAlt = lla.mAlt;
        }
        return *this;
    }

    using LatLon::getLat;
    using LatLon::getLon;
    using LatLon::getLatRadians;
    using LatLon::getLonRadians;
    using LatLon::setLat;
    using LatLon::setLon;
    using LatLon::setLatRadians;
    using LatLon::setLonRadians;

    double getAlt() const
    {
        return mAlt;
    }

    void setAlt(double alt)
    {
        mAlt = alt;
    }

    bool operator==(const LatLonAlt& x) const
    {
        return mLat == x.mLat && mLon == x.mLon && mAlt == x.mAlt;
    }

protected:
    double mAlt;
};

}

#endif


