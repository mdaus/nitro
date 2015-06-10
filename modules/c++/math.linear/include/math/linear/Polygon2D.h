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

#ifndef MATH_LINEAR_POLYGON2D_2_H_
#define MATH_LINEAR_POLYGON2D_2_H_

#define MOE 0.00000000000001

#include <iomanip>
#include <vector>
#include <math/linear/Line2D.h>
#include <algs/ConvexHull.h>

namespace math
{
namespace linear
{
class Polygon2D
{
public:
    Polygon2D(std::vector<Line2D::Point> v)
    {
        algs::ConvexHull<double> polygon(v, points);
        if (getNVerts() < 3)
        {
            throw except::Exception("Need 3 or more valid points to create a Polygon");
        }
    }
    bool pointInPolygon(const Line2D::Point& p) const
    {
        int flag = 0;
        double px = p.row;
        double py = p.col;
        int i, j;

        // If point is a vertex, return true
        for (i = 0; i < getNVerts(); i++)
        {
            double diffRow = std::abs((points.at(i)).row - p.row);
            double diffCol = std::abs((points.at(i)).col - p.col);
            if (diffRow < MOE
             && diffCol < MOE)
            {
                return true;
            }
        }
        // this fails, protects against divide by zero
        for (i = 0, j = getNVerts() - 1; i < getNVerts(); j = i++) {
            if ( ((getY(i) > py) != (getY(j) > py))  &&
                 (px < (getX(j)-getX(i)) * (py-getY(i)) / (getY(j)-getY(i)) + getX(i)) ) {
                flag ^= 1;
            }
        }
        return (flag == 1);
    }
    Line2D::Point getVertex(int i) const
    {
        return points.at(i);
    }
    int getX(int i) const
    {
        return points.at(i).row;
    }
    int getY(int i) const
    {
        return points.at(i).col;
    }
    int getNVerts() const
    {
        // Because the convex hull stores the first point at the
        // beginning and end of the vector 'points', actual size is one less:
        return points.size() - 1;
    }
    bool onSegment(const Line2D::Point& p, const Line2D::Point& end1,
            const Line2D::Point& end2) const
    {
        return !( (p.row  < end1.row  && p.row  < end2.row)  ||
             (p.row  > end1.row  && p.row  > end2.row)  ||
             (p.col < end1.col && p.col < end2.col) ||
             (p.col > end1.col && p.col > end2.col) );
    }

    Polygon2D intersection(const Polygon2D& other) const
    {
        // row find where any line segments intersect - those points are
        // added to the vertex list
        std::vector < Line2D::Point > VertexList;
        for (int i = 0; i < getNVerts(); i++) {
            int P1_next = (i + 1) % getNVerts();

            Line2D P1Segment (points.at(i), points.at(P1_next));

            for (int j = 0; j < other.getNVerts(); j++) {
                int P2_next = (j + 1) % other.getNVerts();
                try {
                    Line2D P2Segment (other.points.at(j), other.points.at(P2_next));
                    Line2D::Point v = P1Segment.intersection(P2Segment);
                    if ( onSegment(v, points.at(i), points.at(P1_next)) &&
                            onSegment(v, other.points.at(j), other.points.at(P2_next)))
                    {
                        VertexList.push_back(v);
                    }
                }
                catch (except::Exception& e)
                {
                    e.getMessage();
                }
            }
        }

        // Next add any vertices of Polygon1 that are contained in other
        std::vector <Line2D::Point > P1_in;
        for (int i = 0; i < getNVerts(); i++) {
            if (other.pointInPolygon(points.at(i)) ) {
                P1_in.push_back(points.at(i));
            }
        }

        // Finally add any vertices of other that are contained in Polygon1
        std::vector <Line2D::Point > P2_in;
        for (int i = 0; i < other.getNVerts(); i++) {
            if (pointInPolygon(other.points.at(i)) ) {
                P2_in.push_back(other.points.at(i));
            }
        }

        // Combine the lists
        for (int i = 0; i < P1_in.size(); i++)
            VertexList.push_back(P1_in.at(i));

        for (int i = 0; i < P2_in.size(); i++)
            VertexList.push_back(P2_in.at(i));
        try
        {
            Polygon2D ret(VertexList);
            return ret;
        }
        catch (except::Exception& e)
        {
            throw except::Exception("Not enough vertices intersect to create a Polygon");
        }
    }
    // for debugging
    void printPoints()
    {
        for (size_t i = 0; i < points.size(); ++i)
        {
            Line2D::Point curr = points[i];
            std::cout << std::setprecision(15) << "(" << curr.row << "," << curr.col << ")" << std::endl;
        }
    }
private:
    std::vector<Line2D::Point> points;
};
}
}

#endif
