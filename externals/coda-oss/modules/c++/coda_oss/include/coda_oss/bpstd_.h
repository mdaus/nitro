/* =========================================================================
 * This file is part of coda_oss-c++
 * =========================================================================
 *
 * (C) Copyright 2020, Maxar Technologies, Inc.
 *
 * sys-c++ is free software; you can redistribute it and/or modify
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
 * License along with this program; If not, http://www.gnu.org/licenses/.
 *
 */
#ifndef CODA_OSS_coda_oss_bpstd__h_INCLUDED_
#define CODA_OSS_coda_oss_bpstd__h_INCLUDED_
#pragma once

#include "coda_oss/CPlusPlus.h"

// Should we use bpstd/ ?
#ifndef CODA_OSS_coda_oss_USE_BPSTD_
	#define CODA_OSS_coda_oss_USE_BPSTD_ !CODA_OSS_cpp20 // yes, use it; not needed at all w/C++20
	//#define CODA_OSS_coda_oss_USE_BPSTD_ 0  // no, use our own
#endif  // CODA_OSS_coda_oss_USE_BPSTD_

#endif  // CODA_OSS_coda_oss_bpstd__h_INCLUDED_
