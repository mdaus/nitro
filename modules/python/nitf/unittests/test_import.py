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

import os

def test_nitf_import():
    import nitf

    print("imported nitf module")
    return True


if __name__ == "__main__":

    if 'DEBUG_PYTHONPATH' in os.environ:
        import sys
        from pprint import pprint
        for dirname in sys.path:
            try:
                print('Contents of %s' % dirname)
                pprint(os.listdir(dirname))
            except:
                pass

    test_nitf_import()
