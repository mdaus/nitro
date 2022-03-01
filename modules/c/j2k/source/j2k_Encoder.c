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
#include "j2k/j2k_Encoder.h"

#ifndef HAVE_OPENJPEG_H

J2KAPI(j2k_codec_t*) j2k_create_compress(void)
{
	return NULL;
}

J2KAPI(void) opj_destroy_codec(j2k_codec_t* pEncoder)
{

}

J2KAPI(j2k_cparameters_t*) j2k_set_default_encoder_parameters(void)
{
	return NULL;
}
J2KAPI(void) j2k_destroy_encoder_parameters(j2k_cparameters_t* pParameters)
{

}
J2KAPI(NRT_BOOL) j2k_initEncoderParameters(j2k_cparameters_t* pParameters,
	size_t tileRow, size_t tileCol, double compressionRatio, size_t numResolutions)
{
	return NRT_FALSE;
}

J2KAPI(NRT_BOOL) j2k_set_error_handler(j2k_codec_t* p_codec, j2k_msg_callback p_callback, void* p_user_data)
{
	return NRT_FALSE;
}

J2KAPI(NRT_BOOL) j2k_setup_encoder(j2k_codec_t* p_codec, const j2k_cparameters_t* parameters, j2k_Image* image)
{
	return NRT_FALSE;
}


#endif