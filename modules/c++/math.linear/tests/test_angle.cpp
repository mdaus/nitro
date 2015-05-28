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

#include <TestCase.h>
#include <import/math/linear.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Explicitly recreating the Vector3D implementation of angle function
double vec3DAngle(const math::linear::VectorN<3>& v1, const math::linear::VectorN<3>& v2)
{
    double val1 = (v1.dot(v2)) / v1.norm() / v2.norm();
    if (val1 > 1.0) val1 = 1.0;
    if (val1 < -1.0) val1 = -1.0;
    return (std::acos(val1));
}
double vecAngle(const math::linear::Vector<double>& v1, const math::linear::Vector<double>& v2)
{
    double val1 = (v1.dot(v2)) / v1.norm() / v2.norm();
    if (val1 > 1.0) val1 = 1.0;
    if (val1 < -1.0) val1 = -1.0;
    return (std::acos(val1));
}
TEST_CASE(randAnglesVN)
{
    for(int i = 0; i < 20000; ++i)
    {
        int i1 = rand() % 1000;
        int i2 = rand() % 1000;
        int i3 = rand() % 1000;
        math::linear::VectorN<3> v1;
        v1[0] = i1; v1[1] = i2; v1[2] = i3;

        int j1 = rand() % 1000;
        int j2 = rand() % 1000;
        int j3 = rand() % 1000;
        math::linear::VectorN<3> v2;
        v2[0] = j1; v2[1] = j2; v2[2] = j3;

        double expected = vec3DAngle(v1, v2);
        double actual = v1.angle(v2);
        TEST_ASSERT_ALMOST_EQ(actual, expected);
    }
}
TEST_CASE(randAnglesV)
{
    for(int i = 0; i < 20000; ++i)
    {
        int i1 = rand() % 1000;
        int i2 = rand() % 1000;
        int i3 = rand() % 1000;
        math::linear::Vector<double> v1(3);
        v1[0] = i1; v1[1] = i2; v1[2] = i3;

        int j1 = rand() % 1000;
        int j2 = rand() % 1000;
        int j3 = rand() % 1000;
        math::linear::Vector<double> v2(3);
        v2[0] = j1; v2[1] = j2; v2[2] = j3;

        double expected = vecAngle(v1, v2);
        double actual = v1.angle(v2);
        TEST_ASSERT_ALMOST_EQ(actual, expected);
    }
}
int main()
{
    TEST_CHECK(randAnglesVN);
    TEST_CHECK(randAnglesV);
    return 0;
}
