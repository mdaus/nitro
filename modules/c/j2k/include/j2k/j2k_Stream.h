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
#ifndef NITRO_j2k_Strean_h_INCLUDED_ 
#define NITRO_j2k_Strean_h_INCLUDED_

#include "j2k/Defines.h"

J2K_CXX_GUARD

typedef struct _j2k_Stream
{
    void /*obj_stream_t*/ *opj_stream;
} j2k_Stream;

#define NITRO_J2K_STREAM_CHUNK_SIZE 0x100000 /** 1 mega by default */ // c.f. OPJ_J2K_STREAM_CHUNK_SIZE in <openjpeg.h>

J2KAPI(j2k_Stream*) j2k_Stream_create(size_t chunkSize, J2K_BOOL isInputStream);
J2KAPI(void) j2k_Stream_destroy(j2k_Stream* pStream);

J2K_CXX_ENDGUARD

#endif // NITRO_j2k_Strean_h_INCLUDED_

