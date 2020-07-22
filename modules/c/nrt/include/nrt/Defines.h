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

#ifndef __NRT_DEFINES_H__
#define __NRT_DEFINES_H__
#pragma once

#ifdef _MSC_VER
#pragma warning(disable: 4668) // '...' is not defined as a preprocessor macro, replacing with '...' for '...'
#pragma warning(disable: 4820) // '...': '...' bytes padding added after data member '...'
#pragma warning(disable: 4710) // '...': function not inlined
#pragma warning(disable: 4255) // '...': no function prototype given : converting '...' to '...'

#pragma warning(disable: 5045) // Compiler will insert Spectre mitigation for memory load if / Qspectre switch specified
#endif

#ifdef _MSC_VER // TODO: get rid of these someday?
#pragma warning(disable: 4100) // '...': unreferenced formal parameter
#pragma warning(disable: 4296) // '...': expression is always false
#pragma warning(disable: 4267) // '...': conversion from '...' to '...', possible loss of data
#pragma warning(disable: 4244) // 	'...': conversion from '...' to '...', possible loss of data
#pragma warning(disable: 4242) // '...': conversion from '...' to '...', possible loss of data
#pragma warning(disable: 4018) // '...': signed / unsigned mismatch
#pragma warning(disable: 4389) // '...': signed / unsigned mismatch
#endif

/* The version of the NRT library */

#ifdef __cplusplus
#   define NRT_C extern "C"
#   define NRT_GUARD {
#   define NRT_ENDGUARD }
#   define NRT_BOOL bool
#else
#   define NRT_C
#   define NRT_GUARD
#   define NRT_ENDGUARD
#   define NRT_BOOL int
#endif

#ifdef WIN32
/*  Negotiate the meaning of NRTAPI, NRTPROT (for public and protected)  */
#      if defined(NRT_MODULE_EXPORTS)
#          define NRTAPI(RT)  NRT_C __declspec(dllexport) RT
#          define NRTPROT(RT) NRT_C __declspec(dllexport) RT
#      elif defined(NRT_MODULE_IMPORTS)
#          define NRTAPI(RT)  NRT_C __declspec(dllimport) RT
#          define NRTPROT(RT) NRT_C __declspec(dllexport) RT
#      else                     /* Static library */
#          define NRTAPI(RT) NRT_C RT
#          define NRTPROT(RT) NRT_C RT
#      endif
/*  String conversion, on windows  */
#      define  NRT_ATO32(A) strtol(A, (char **)NULL, 10)
#      define  NRT_ATOU32(A) strtoul(A, (char **)NULL, 10)
#      define  NRT_ATOU32_BASE(A,B) strtoul(A, (char **)NULL, B)
#      define  NRT_ATO64(A) _atoi64(A)
#      define  NRT_SNPRINTF _snprintf
#      define  NRT_VSNPRINTF _vsnprintf
#else
/*
*  NRTAPI and NRTPROT don't mean as much on Unix since they
*  Don't try and mangle functions for you in C
*/
#      define  NRTAPI(RT)  NRT_C RT
#      define  NRTPROT(RT) NRT_C RT
#      define  NRT_ATO32(A) strtol(A, (char **)NULL, 10)
#      define  NRT_ATOU32(A) strtoul(A, (char **)NULL, 10)
#      define  NRT_ATOU32_BASE(A,B) strtoul(A, (char **)NULL, B)
#      if defined(__aix__)
#          define  NRT_ATO64(A) strtoll(A, 0, 0)
#      else
#          define  NRT_ATO64(A) atoll(A)
#      endif
#      define NRT_SNPRINTF snprintf
#      define NRT_VSNPRINTF vsnprintf
#endif
/*
 *  This section describes a set of macros to help with
 *  C++ compilation.  The 'extern C' set is required to
 *  prevent name-mangling.
 *
 *  We also define a boolean according to the language,
 *  since it is obnoxious to use an int when C++ has a
 *  perfectly good one already.
 */

#define NRT_CXX_GUARD     NRT_C NRT_GUARD
#define NRT_CXX_ENDGUARD  NRT_ENDGUARD

/*  Private declarations.  */
#define NRTPRIV(RT) static RT

/*  Negotiate the 'context'  */
#define NRT_FILE __FILE__
#define NRT_LINE __LINE__
#if defined(__GNUC__)
#    define NRT_FUNC __PRETTY_FUNCTION__
#elif __STDC_VERSION__ < 199901
#    define NRT_FUNC "unknown function"
#else                           /* Should be c99 */
#    define NRT_FUNC __func__
#endif

#endif
