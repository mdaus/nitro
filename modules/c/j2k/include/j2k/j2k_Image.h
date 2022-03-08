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

 // J2K isn't part of "nitf" (yet?) so use NITRO, not NITF prefix
#ifndef NITRO_j2k_Image_h_INCLUDED_ 
#define NITRO_j2k_Image_h_INCLUDED_

#include "j2k/Defines.h"

J2K_CXX_GUARD

typedef struct _j2k_image_t
{
    void /*opj_image_t*/* opj_image;
} j2k_image_t;

typedef struct _j2k_image_comptparm // c.f. opj_image_comptparm in <openjpeg.h>
{
    uint32_t dx;     /** XRsiz: horizontal separation of a sample of ith component with respect to the reference grid */
    uint32_t dy;     /** YRsiz: vertical separation of a sample of ith component with respect to the reference grid */
    uint32_t w;     /** data width */
    uint32_t h;     /** data height */
    uint32_t x0;     /** x component offset compared to the whole image */
    uint32_t y0;     /** y component offset compared to the whole image */
    uint32_t prec;     /** precision */
    uint32_t bpp;  /** image depth in bits */
    J2K_BOOL sgnd;     /** signed (1) / unsigned (0) */
} j2k_image_comptparm;

typedef enum J2K_COLOR_SPACE { // c.f. OBJ_COLOR_SPACE in <openjpeg.h>
    J2K_CLRSPC_UNKNOWN = -1,    /**< not supported by the library */
    J2K_CLRSPC_UNSPECIFIED = 0, /**< not specified in the codestream */
    J2K_CLRSPC_SRGB = 1,        /**< sRGB */
    J2K_CLRSPC_GRAY = 2,        /**< grayscale */
    J2K_CLRSPC_SYCC = 3,        /**< YUV */
    J2K_CLRSPC_EYCC = 4,        /**< e-YCC */
    J2K_CLRSPC_CMYK = 5         /**< CMYK */
} J2K_COLOR_SPACE;

J2KAPI(j2k_image_t*) j2k_image_tile_create(uint32_t numcmpts, const j2k_image_comptparm* cmptparms, J2K_COLOR_SPACE clrspc);
J2KAPI(J2K_BOOL) j2k_image_init(j2k_image_t* pImage, int x0, int y0, int x1, int y1, int numcmpts, J2K_COLOR_SPACE color_space);
J2KAPI(void) j2k_image_destroy(j2k_image_t* pImage);

J2K_CXX_ENDGUARD

#endif // NITRO_j2k_Image_h_INCLUDED_
