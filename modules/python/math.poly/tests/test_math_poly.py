#!/usr/bin/env python

"""
 * =========================================================================
 * This file is part of math.poly-c++ 
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
 *
"""

import sys
from math_poly import *

if __name__ == '__main__':
    #################
    # Basic 1D test #
    #################

    poly1D = Poly1D(2)
    for x in range(poly1D.order() + 1):
        poly1D[x] = (x + 1) * 10

    print '1D poly:'
    print poly1D

    # Try to index out of bounds by getting
    threw = False
    try:
        foo = poly1D[3]
    except ValueError:
        threw = True

    if threw:
        print 'Getting 1D OOB threw as expected'
    else:
        sys.exit('Getting 1D OOB did not throw!')

    # Try to index out of bounds by setting
    threw = False
    try:
        poly1D[3] = 5
    except ValueError:
        threw = True

    if threw:
        print 'Setting 1D OOB threw as expected'
    else:
        sys.exit('Setting 1D OOB did not throw!')

    #################
    # Basic 2D test #
    #################

    poly2D = Poly2D(2, 3)
    val = 100
    for x in range(poly2D.orderX() + 1):
        for y in range(poly2D.orderY() + 1):
            poly2D[(x, y)] = val
            val += 100

    print '\n2D poly:'
    print poly2D

    # Try to index out of bounds by getting
    threw = False
    try:
        foo = poly2D[(3, 3)]
    except ValueError:
        threw = True

    if threw:
        print 'Getting 2D OOB threw as expected'
    else:
        sys.exit('Getting 2D OOB did not throw!')

    # Try to index out of bounds by setting
    threw = False
    try:
        poly2D[(3, 3)] = 5
    except ValueError:
        threw = True

    if threw:
        print 'Setting 2D OOB threw as expected'
    else:
        sys.exit('Setting 2D OOB did not throw!')
