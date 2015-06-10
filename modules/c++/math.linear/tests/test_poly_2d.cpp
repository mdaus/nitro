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

#include <math/linear/Polygon2D.h>
#include <TestCase.h>

namespace
{
TEST_CASE(testBelowThreePts)
{
    math::linear::Line2D::Point P1(2, 4);
    math::linear::Line2D::Point P2(4, 3);
    std::vector<math::linear::Line2D::Point> v;
    v.push_back(P1);
    v.push_back(P2);
    TEST_EXCEPTION(math::linear::Polygon2D(v));
}
TEST_CASE(testAtThreePts)
{
    math::linear::Line2D::Point P1(2, 4);
    math::linear::Line2D::Point P2(4, 3);
    math::linear::Line2D::Point P3(7, 6);
    std::vector<math::linear::Line2D::Point> v;
    v.push_back(P1);
    v.push_back(P2);
    v.push_back(P3);
    math::linear::Polygon2D poly(v);
    TEST_ASSERT(poly.pointInPolygon(P1));
    TEST_ASSERT(poly.pointInPolygon(P2));
    TEST_ASSERT(poly.pointInPolygon(P3));
    TEST_ASSERT_EQ(poly.getNVerts(), 3);

    TEST_ASSERT(poly.onSegment(P1, P1, P2));
    TEST_ASSERT(poly.onSegment(P2, P1, P2));
    TEST_ASSERT(!(poly.onSegment(P1, P2, P3)));
}
TEST_CASE(testJunkPts)
{
    math::linear::Line2D::Point P1(2, 4);
    math::linear::Line2D::Point P2(4, 3);
    math::linear::Line2D::Point P3(7, 6);
    std::vector<math::linear::Line2D::Point> v;
    v.push_back(P1);
    v.push_back(P2);
    v.push_back(P3);
    // Throwing some redundant vertices in to make sure Polygon2D removes them
    math::linear::Line2D::Point P4 = math::linear::Line2D(P1, P2).offsetFromPoint(P1, 1);
    v.push_back(P4);
    v.push_back(P2);

    math::linear::Polygon2D poly(v);
    // should only have 3 real vertices
    TEST_ASSERT_EQ(poly.getNVerts(), 3);

    std::vector<math::linear::Line2D::Point> v2;
    v2.push_back(P1);
    v2.push_back(P2);
    v2.push_back(P4);
    // should throw an exception since this only has 2 valid points (need 3)
    TEST_EXCEPTION(math::linear::Polygon2D(v2));
}
}

int main()
{
    TEST_CHECK(testBelowThreePts);
    TEST_CHECK(testAtThreePts);
    TEST_CHECK(testJunkPts);
    return 0;
}
