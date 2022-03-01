/* =========================================================================
 * This file is part of NITRO
 * =========================================================================
 *
 * (C) Copyright 2022, Maxar Technologies, Inc.
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

#include "j2k/Config.h"
#include "j2k/j2k_Image.h"

#ifndef HAVE_OPENJPEG_H
J2KAPI(j2k_Image*) j2k_Image_tile_create(uint32_t numcmpts, const j2k_Image_comptparm* cmptparms, J2K_COLOR_SPACE clrspc)
{
	return NULL;
}

J2KAPI(J2K_BOOL) j2k_Image_init(j2k_Image* pImage, int x0, int y0, int x1, int y1, int numcmpts, J2K_COLOR_SPACE color_space)
{
	return J2K_FALSE;
}

J2KAPI(void) j2k_Image_destroy(j2k_Image* pImage)
{
}
#endif