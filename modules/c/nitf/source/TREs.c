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

#undef NITF_PLUGIN_FUNCTION_EXPORT
#define NITF_PLUGIN_FUNCTION_EXPORT(retval_) static retval_

#include "nitf/TRE.h"
#include "nitf/PluginIdentifier.h"

#if _MSC_VER
#pragma warning(disable: 4464) // relative include path contains '..'
#endif

#include "../shared/ACCHZB.c"
#include "../shared/HISTOA.c"
#include "../shared/TEST_DES.c"
#include "../shared/XML_DATA_CONTENT.c"

extern const nitf_TREPreloaded preloaded[];
const nitf_TREPreloaded preloaded[] = {
	{ "ACCHZB",  ACCHZB_init, ACCHZB_handler },
	{ "HISTOA",  HISTOA_init, HISTOA_handler },
	{ NULL, NULL, NULL }
};
