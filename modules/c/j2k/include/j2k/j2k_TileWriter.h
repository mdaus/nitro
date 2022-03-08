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

 // J2K isn't part of "nitf" (yet?) so use NITRO, not NITF prefix
#ifndef NITRO_j2k_TileWriter_h_INCLUDED_ 
#define NITRO_j2k_TileWriter_h_INCLUDED_

#include "j2k/j2k_Encoder.h"
#include "j2k/j2k_Image.h"

J2K_CXX_GUARD

typedef struct _j2k_stream_t
{
    void /*obj_stream_t*/* opj_stream;
} j2k_stream_t;

#define NITRO_J2K_STREAM_CHUNK_SIZE 0x100000 /** 1 mega by default */ // c.f. OPJ_J2K_STREAM_CHUNK_SIZE in <openjpeg.h>

J2KAPI(j2k_stream_t*) j2k_stream_create(size_t chunkSize, J2K_BOOL isInputStream);
J2KAPI(void) j2k_stream_destroy(j2k_stream_t* pStream);

//----------------------------------------------------------------------------------------------------------------

typedef size_t(*j2k_stream_write_fn)(void* p_buffer, size_t p_nb_bytes, void* p_user_data);
typedef int64_t(*j2k_stream_skip_fn)(int64_t p_nb_bytes, void* p_user_data);
typedef NRT_BOOL(*j2k_stream_seek_fn)(int64_t p_nb_bytes, void* p_user_data);
typedef void (*j2k_stream_free_user_data_fn)(void* p_user_data);

J2KAPI(void) j2k_stream_set_write_function(j2k_stream_t* p_stream, j2k_stream_write_fn p_function);
J2KAPI(void) j2k_stream_set_skip_function(j2k_stream_t* p_stream, j2k_stream_skip_fn p_function);
J2KAPI(void) j2k_stream_set_seek_function(j2k_stream_t* p_stream, j2k_stream_seek_fn p_function);

J2KAPI(void) j2k_stream_set_user_data(j2k_stream_t* p_stream, void* p_data, j2k_stream_free_user_data_fn p_function);

J2KAPI(NRT_BOOL) j2k_flush(j2k_codec_t* p_codec, j2k_stream_t* p_stream);
J2KAPI(NRT_BOOL) j2k_start_compress(j2k_codec_t* p_codec, j2k_image_t* p_image, j2k_stream_t* p_stream);
J2KAPI(NRT_BOOL) j2k_end_compress(j2k_codec_t* p_codec, j2k_stream_t* p_stream);

J2KAPI(NRT_BOOL) j2k_write_tile(j2k_codec_t* p_codec, uint32_t p_tile_index, const uint8_t* p_data, uint32_t p_data_size, j2k_stream_t* p_stream);

J2K_CXX_ENDGUARD

#endif // NITRO_j2k_TileWriter_h_INCLUDED_
