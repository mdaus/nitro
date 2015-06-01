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

#ifndef __MATH_LINEAR_LINE_2D_H__
#define __MATH_LINEAR_LINE_2D_H__

#define MOE 0.00000000001 // margin of error--yet to be confirmed


namespace math
{
namespace linear
{
typedef types::RowCol<double> Point2D;

class Line2D
{

public:
    enum Line2DType { INVALID, NORMAL, HORIZONTAL, VERTICAL };
    Line2D() : slope(0), yIntercept(0), xIntercept(0)
    {
        type = Line2D::INVALID;
    }
    Line2D(const Line2D& other) :slope(other.slope), xIntercept(other.xIntercept),
            yIntercept(other.yIntercept), type(Line2D::INVALID)
    {

    }
    Line2D(Point2D P1, Point2D P2)
    {
        double dx = P2.row - P1.col;
        double dy = P2.col - P1.col;

        // error handling: what should we do here?
        if ((std::abs(dy) < MOE) && (std::abs(dx) < MOE))
        {
            type = Line2D::INVALID;
            return;
        }
        //Vertical if x values are the same
         if (std::abs(dx) < MOE)
         {
             type = Line2D::VERTICAL;
             slope = 0;
             xIntercept = P1.row;
             return;
         }
        // Horizontal if y values are the same
        if (std::abs(dy) < MOE)
        {
            type = Line2D::HORIZONTAL;
            slope = 0;
            yIntercept = P1.col;
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
        type = Line2D::NORMAL;
        if (std::abs(slope) < MOE)
        {
            yIntercept = P.col; // xIntercept not intialized?
            return;
        }
        yIntercept = P.col - P.col * slope; // fix
        xIntercept = x(0.0);
    }
    ~Line2D()
    {
        // Why does this need to be here if it doesn't do anything? Style?
    }
    double getSlope() { return slope; }
    double getYIntercept() { return yIntercept; }
    //Evaluate for y given x:
    double y(double x)
    {
        if (type == Line2D::INVALID)
        {
            // throw some error
            return 0;
        }
        if (type == Line2D::VERTICAL)
        {
            // Check that x is not the xIntercept
            if (std::abs(x - xIntercept) > MOE)
            {
                // throw some error
            }
            // throw a different error
            return 0;
        }
        if (type == Line2D::HORIZONTAL)
        {
            return yIntercept;
        }
        return slope * x + yIntercept;
    }
    //Evaluate for x given y:
    double x(double y)
    {
        if (type == Line2D::INVALID)
        {
            //throw some error
            return 0;
        }
        if (type == Line2D::HORIZONTAL)
        {
            //check for y being the y intercept
            if (std::abs(y - yIntercept) > MOE)
            {
                //throw some error
            }
            // throw a different error
            return 0;
        }
        if (type == Line2D::VERTICAL)
        {
            return xIntercept;
        }
        return (y - yIntercept) / slope;
    }

    // Error handling with this function? Throw exception if they do not intersect? return uninitialized Point2D?
    Point2D intersection(Line2D L, int*err)
    {
        Point2D P(0.0, 0.0);
        //check for error conditions:
        if (type == Line2D::INVALID || L.type == Line2D::INVALID)
        {
            //throw error?
            return P;
        }
        // if lines vertical parallel:
        if (type == Line2D::VERTICAL && L.type == Line2D::VERTICAL)
        {
            // throw error?
            return P;
        }
        // if lines parallel
        if (slope == L.slope)
        {
            //throw error?
            return P;
        }
        // Calculate intersection
        if (type == Line2D::VERTICAL)
        {
            // First line is vertical
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
        *err = 0;
        return P;
    }

    // Create a new line parallel to this line through point P
    Line2D parallelToLine(Point2D P)
    {
        if (type == Line2D::INVALID) { return Line2D();} // What why? find a better solution here
        if (type == Line2D::VERTICAL)
        {
            // create a new vertical line through our point
            Point2D P2 = P;
            P2.col += 1; // offset in y---why is this 1 ? Maybe add a parameter for the user to enter how far away they want it
            return Line2D(P, P2);
        }
        // other lines can just take the slope
        return Line2D(P, slope);
    }
    // Create a new line perpendicular to this line through point P
    Line2D perpindicularToLine(Point2D P)
    {
        if (type == Line2D::INVALID) { return Line2D(); } // Maybe find a better solution here
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
        return Line2D(P, (-1.0 / slope)); // how is the second argument a point?
    }

    // Compute distance from this line to a point--error handling?
    //might be able to implement this with some CODA stuff
    double distanceToPoint(Point2D P)
    {
        if (type == Line2D::INVALID)
        {
            //throw error?
            return 0;
        }
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

    //Return a point that is a dsitance d from the point P which is on the line
    //Error handling? This seems like a throwaway fxn
    Point2D offsetFromPoint(Point2D P, double distance)
    {
        if (type == Line2D::INVALID)
        {
            //throw some error
            return P;
        }
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
