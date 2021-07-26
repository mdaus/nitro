#!/usr/bin/env python3

"""
 * =========================================================================
 * This file is part of NITRO
 * =========================================================================
 * 
 * (C) Copyright 2004 - 2021, MDA Information Systems LLC
 *
 * NITRO is free software; you can redistribute it and/or modify
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
 * License along with this program; if not, If not, 
 * see <http://www.gnu.org/licenses/>.
 *
 *
"""


import sys

global_failure = False


def ok(ans):
    if not ans:
        global_failure = True


def complete():
    if global_failure:
        sys.exit(-1)


def test_nitf_import():
    try:
        import nitf

        print("imported nitf module")
        return True
    except e as Exception:
        print("failed to import nitf module")
        return False


if __name__ == "__main__":
    ok(test_nitf_import())
