/* =========================================================================
 * This file is part of math.linear-c++ 
 * =========================================================================
 * 
 * (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
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
#include <import/math/linear.h>

namespace mx=math::linear;

typedef mx::MatrixMxN<2, 2> _2x2;
typedef mx::MatrixMxN<3, 2> _3x2;
typedef mx::MatrixMxN<3, 3> _3x3;
typedef mx::MatrixMxN<4, 4> _4x4;


int main()
{
    _3x3 A = mx::identityMatrix<3, double>();
    std::cout << A << std::endl;

    _3x2 B = mx::constantMatrix<3, 2, double>(1);
 
    B.scale(0.5);
    std::cout << B << std::endl;
   
    _3x2 C = A * B;
    std::cout << C << std::endl;

    _2x2 D = mx::constantMatrix<2, 2, double>(1);
    D(0, 0) = 2;
    D(1, 1) = 3;

    // 2 1
    // 1 3
    std::cout << D << std::endl;
    
    _2x2 E = mx::constantMatrix<2, 2, double>(2);

    // 3 2
    // 2 2
    
    E(0, 0) = 3;

    // 6 + 2 4 + 2
    // 3 + 6 2 + 6
    
    _2x2 F = mx::identityMatrix<2, double>();

    std::cout << (D * E) - F << std::endl;


    int p[2] = { 1, 0 };
    _2x2 G = F.permute(p);
    std::cout << G << std::endl;

    std::cout << "Inv: " << std::endl;
    std::cout << mx::inverse<2, 2, double>(G) << std::endl;

    _3x3 H = mx::identityMatrix<3, double>();

    H(0, 1) = 0.2;
    std::cout << "H: " << H << std::endl;
    

    std::cout << "H inv: " << mx::inverse(H) << std::endl;
    
}
