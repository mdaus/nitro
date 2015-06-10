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

TEST_CASE(testNonIntersecting)
{
    math::linear::Line2D::Point P1(2,1);
    math::linear::Line2D::Point P2(3,4);
    math::linear::Line2D::Point P3(5,2);
    math::linear::Line2D::Point P4(6,6);
    math::linear::Line2D::Point P5(8,4);
    std::vector<math::linear::Line2D::Point> v;
    v.push_back(P1);
    v.push_back(P2);
    v.push_back(P3);
    v.push_back(P4);
    v.push_back(P5);
    math::linear::Polygon2D poly1(v);

    math::linear::Line2D::Point P6(6,1);
    math::linear::Line2D::Point P7(8,3);
    math::linear::Line2D::Point P8(9,1);
    std::vector<math::linear::Line2D::Point> v2;
    v2.push_back(P6);
    v2.push_back(P7);
    v2.push_back(P8);
    math::linear::Polygon2D poly2(v2);

    // expect an exception thrown since they do not intersect
    TEST_EXCEPTION(poly1.intersection(poly2));
}
TEST_CASE(testIntersectingOnePt)
{

    math::linear::Line2D::Point P1(2,1);
    math::linear::Line2D::Point P2(3,4);
    math::linear::Line2D::Point P3(5,2);
    math::linear::Line2D::Point P4(6,6);
    math::linear::Line2D::Point P5(8,4);
    std::vector<math::linear::Line2D::Point> v;
    v.push_back(P1);
    v.push_back(P2);
    v.push_back(P3);
    v.push_back(P4);
    v.push_back(P5);
    math::linear::Polygon2D poly1(v);

    math::linear::Line2D::Point P6(11,2);
    math::linear::Line2D::Point P7(11,7);
    std::vector<math::linear::Line2D::Point> v2;
    v2.push_back(P5);
    v2.push_back(P6);
    v2.push_back(P7);
    math::linear::Polygon2D poly2(v2);

    // expect an exception thrown since you can't make an intersecting
    // polygon when they only intersect with one point
    TEST_EXCEPTION(poly1.intersection(poly2));
}
TEST_CASE(testIntersecting)
{
    // observe
    math::linear::Line2D::Point P1(2,1);
    math::linear::Line2D::Point P2(3,4);
    math::linear::Line2D::Point P3(5,2);
    math::linear::Line2D::Point P4(6,6);
    math::linear::Line2D::Point P5(8,4);
    std::vector<math::linear::Line2D::Point> v;
    v.push_back(P1);
    v.push_back(P2);
    v.push_back(P3);
    v.push_back(P4);
    v.push_back(P5);
    math::linear::Polygon2D poly1(v);

    math::linear::Line2D::Point P6(4,4);
    math::linear::Line2D::Point P7(6,4);
    math::linear::Line2D::Point P8(8,8);
    math::linear::Line2D::Point P9(4,6);
    std::vector<math::linear::Line2D::Point> v2;
    v2.push_back(P6);
    v2.push_back(P7);
    v2.push_back(P8);
    v2.push_back(P9);
    math::linear::Polygon2D poly2(v2);

    math::linear::Polygon2D iPoly = poly1.intersection(poly2);
    math::linear::Line2D::Point exP1(4,4);
    math::linear::Line2D::Point exP2(6,4);
    math::linear::Line2D::Point exP3(4,4.66666666666667);
    math::linear::Line2D::Point exP4(6,6);
    math::linear::Line2D::Point exP5(6.66666666666667, 5.33333333333333);
    TEST_ASSERT(iPoly.pointInPolygon(exP1));
    TEST_ASSERT(iPoly.pointInPolygon(exP2));
    TEST_ASSERT(iPoly.pointInPolygon(exP3));
    TEST_ASSERT(iPoly.pointInPolygon(exP4));
    TEST_ASSERT(iPoly.pointInPolygon(exP5));
}
int main()
{
    TEST_CHECK(testIntersecting);
    TEST_CHECK(testNonIntersecting);
    TEST_CHECK(testIntersectingOnePt);
}
