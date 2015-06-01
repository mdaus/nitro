/* =========================================================================
 * This file is part of math.linear-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2014, MDA Information Systems LLC
 *
 * math.linear-c++ is free software; you can redistribute it and/or modify
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
#include <math/linear/Line2D.h>
#include <vector>

namespace math
{
namespace linear
{
class Polygon2D
{
public:
    Polygon2D()
    {

    }
    Polygon2D(const Polygon2D& other)
    {

    }
    Polygon2D(const std::vector<math::linear::Point2D>& v)
    {

    }
    Polygon2D intersection(const Polygon2D& other) const
    {
        return other;//fixme
    }
    bool pointInPolygon(const Point2D& p) const
    {
        return false; //fixme
    }
    void removeOrderedDuplicateVertices(double p)
    {

    }
    math::linear::Point2D getVertex(int i) const
    {
        return math::linear::Point2D();//fixme

    }
    int getX(int i) const
    {
        // I am guessing this returns the X value for a given Y value on the polygon?
        return 0; //fixme
    }
    int getY(int i) const
    {
        return 0; //fixme

    }
    int getNVerts() const
    {
        //I am guessing this returns the number of vertices
        return 0; //fixme

    }
    Polygon2D& operator=(const Polygon2D& other)
    {
        return *this; //fixme

    }


};
}
}

