/* =========================================================================
 * This file is part of NITRO
 * =========================================================================
 *
 * (C) Copyright 2004 - 2014, MDA Information Systems LLC
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
 */

#include <import/nitf.h>

NITF_CXX_GUARD

static nitf_TREDescription BCKGDA_description[] = {
    {NITF_BCS_N, 8, "Background Width", "BGWIDTH" },
    {NITF_BCS_N, 8, "Background Height", "BGHEIGHT" },
    {NITF_BCS_N, 8, "Background Red", "BGRED" },
    {NITF_BCS_N, 8, "Background Green", "BGGREEN" },
    {NITF_BCS_N, 8, "Background Blue", "BGBLUE" },
    {NITF_BCS_N, 8, "Pixel Size", "PIXSIZE" },
    {NITF_BCS_N, 8, "Pixel Units", "PIXUNITS" },
    {NITF_END, 0, NULL, NULL}
};

NITF_DECLARE_SINGLE_PLUGIN_SIMPLE(BCKGDA)

NITF_CXX_ENDGUARD
