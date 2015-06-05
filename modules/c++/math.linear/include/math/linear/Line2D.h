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
#include <types/RowCol.h>
#include <cmath>
#include <except/Exception.h>
#include <iostream>

#ifndef __MATH_LINEAR_LINE_2D_H__
#define __MATH_LINEAR_LINE_2D_H__

namespace math
{
namespace linear
{
typedef types::RowCol<double> Point2D;

class Line2D
{
public:
    enum Line2DType { NORMAL, HORIZONTAL, VERTICAL };

    Line2D(Point2D P1, Point2D P2)
    {
        double dx = P2.row - P1.row;
        double dy = P2.col - P1.col;

        // error handling: what should we do here?
        if ((dy == 0) && (dx == 0))
        {
            throw except::Exception("Cannot create a line when P1 == P2");
        }
        //Vertical if x values are the same
        if (dx == 0)
         {
             type = Line2D::VERTICAL;
             slope = 0; // undefined
             xIntercept = P1.row;
             yIntercept = 0; // undefined
             return;
        }
        // Horizontal if y values are the same
        if (dy == 0)
        {
            type = Line2D::HORIZONTAL;
            slope = 0;
            yIntercept = P1.col;
            xIntercept = 0; // undefined
            return;
        }
        type = Line2D::NORMAL;
        slope = dy / dx;
        yIntercept = P1.col - P1.row * slope; // Maybe a better way to do this. go back
        xIntercept = x(0.0);
        return;

    }

    Line2D(Point2D P, double s): slope(s)
    {
        if (slope == 0)
        {
            type = Line2D::HORIZONTAL;
            yIntercept = P.col;
            xIntercept = 0; //undefined
            return;
        }
        type = Line2D::NORMAL;
        yIntercept = P.col - P.row * slope;
        xIntercept = x(0.0);
    }

    double getSlope() const
    {
        if (type == Line2D::VERTICAL)
        {
            throw except::Exception("Vertical line, slope is undefined");
        }
        return slope;
    }

    double getYIntercept() const
    {
        if (type == Line2D::VERTICAL)
        {
            throw except::Exception("No return value for a vertical line with undefined yIntercept");
        }
        return yIntercept;
    }

    double getXIntercept() const
    {
        if (type == Line2D::HORIZONTAL)
        {
            throw except::Exception("No return value for a horizontal line with undefined xIntercept");
        }
        return xIntercept;
    }

    //Evaluate for y given x:
    double y(double x) const
    {
        if (type == Line2D::VERTICAL)
        {
            throw except::Exception("Vertical line--cannot return a single y for given x");
        }
        if (type == Line2D::HORIZONTAL)
        {
            return yIntercept;
        }
        return slope * x + yIntercept;
    }

    //Evaluate for x given y:
    double x(double y) const
    {
        if (type == Line2D::HORIZONTAL)
        {
            throw except::Exception("Horizontal line--cannot return a single x for given y");
        }
        if (type == Line2D::VERTICAL)
        {
            return xIntercept;
        }
        return (y - yIntercept) / slope;
    }

    // Determine intersection of two lines
    Point2D intersection(const Line2D& L) const
    {
        Point2D P(0.0, 0.0);
        if ((slope == L.slope) && (type == L.type))
        {
            throw except::Exception("Two parallel lines--no intersecting point");
        }
        if (type == Line2D::VERTICAL)
        {
            P.row = xIntercept;
            P.col = L.y(P.row);
        }
        else if (L.type == Line2D::VERTICAL)
        {
            P.row = L.xIntercept;
            P.col = y(P.row);
        }
        else
        {
            P.row = (L.yIntercept - yIntercept) / (slope - L.slope);
            P.col = y(P.row);
        }
        return P;
    }

    // Create a new line parallel to this line through point P
    Line2D parallelToLine(Point2D P) const
    {
        if (type == Line2D::VERTICAL)
        {
            // create a new vertical line through our point
            Point2D P2 = P;
            P2.col += 1;
            return Line2D(P, P2);
        }
        // other lines can just take the slope
        return Line2D(P, slope);
    }

    // Create a new line perpendicular to this line through point P
    Line2D perpendicularToLine(Point2D P) const
    {
        if (type == Line2D::HORIZONTAL)
        {
            // create a new vertical line through point P
            Point2D P2 = P;
            P2.col += 1; // offset in y
            return Line2D(P, P2);
        }
        if (type == Line2D::VERTICAL)
        {
            // create a new horizontal line through point P
            Point2D P2 = P;
            P2.row += 1; // offset in x
            return Line2D(P, P2);
        }
        //Other lines can be created from the orthogonal slope and the point
        return Line2D(P, (-1.0 / slope));
    }

    // Compute the distance from this line to a point
    double distanceToPoint(Point2D P) const
    {
        if (type == Line2D::HORIZONTAL)
        {
            return std::abs(P.col - yIntercept);
        }
        if (type == Line2D::VERTICAL)
        {
            return std::abs(P.row - xIntercept);
        }
        const double dist =
                    std::abs(slope * P.row - P.col + yIntercept) /
                    std::sqrt(slope * slope + 1);
        return dist;
    }

    //Return a point that is a distance d from the point P which is on the line
    Point2D offsetFromPoint(Point2D P, double distance) const
    {
        if (type == Line2D::HORIZONTAL)
        {
            P.row += distance;
            return P;
        }
        if (type == Line2D::VERTICAL)
        {
            P.col += distance;
            return P;
        }
        double theta = std::atan(slope);
        P.row += distance * std::cos(theta);
        P.col += distance * std::sin(theta);
        return P;
    }
private:
    Line2DType type;
    double slope;
    double yIntercept;
    double xIntercept;
};
}
}
#endif
