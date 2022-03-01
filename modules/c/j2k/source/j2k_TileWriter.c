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
#include "j2k/j2k_TileWriter.h"

#ifndef HAVE_OPENJPEG_H

J2KAPI(void) j2k_stream_set_write_function(j2k_Stream* p_stream, j2k_stream_write_fn p_function)
{
}

J2KAPI(void) j2k_stream_set_skip_function(j2k_Stream* p_stream, j2k_stream_skip_fn p_function)
{
}

J2KAPI(void) j2k_stream_set_seek_function(j2k_Stream* p_stream, j2k_stream_seek_fn p_function)
{
}

#endif