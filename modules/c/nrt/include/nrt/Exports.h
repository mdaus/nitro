/* =========================================================================
 * This file is part of NITRO
 * =========================================================================
 *
 * (C) Copyright 2004 - 2014, MDA Information Systems LLC
 * © Copyright 2023, Maxar Technologies, Inc.
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

#pragma once

// Need to specify how this code will be consumed, either NITRO_NRT_LIB (static library)
// or NITRO_NRT_DLL (aka "shared" library).  For DLLs, it needs to be set for BOTH
// "exporting" (building this code) and "importing" (consuming).
//
// Use Windows naming conventions (DLL, LIB) because this really only matters for _MSC_VER, see below.
#if !defined(NITRO_NRT_LIB) && !defined(NITRO_NRT_DLL)
    // **A huge HACK so this file can be commited.**
    // Windows clients must know whether the LIB is an static LIB or an export
    // LIB for a DLL (see comments below); thus, CODA_OSS_DLL/CODA_OSS_LIB
    // must always be set to the same value.  Having to add this to every
    // **vcxproj** project is a pain (and error-prone) ... so, the hack is to 
    // use *_FILE_OFFSET_BITS* and *_LARGEFILE_SOURCE* which are
    // only set by our **waf** and/or **cmake** build tools.
    #if defined(_FILE_OFFSET_BITS) || defined(_LARGEFILE_SOURCE)
        // **cmake** or **waf** build, assume static LIB
        #define NITRO_NRT_LIB 1  // Static library, all symbols visible.
    #else
        // Not building with **cmake** or **waf**, assume we're in Visual Studio
        // and thus are using DLLs.
        #define NITRO_NRT_DLL 1  // Symbols must be exported and imported (see below).
    #endif // defined(_FILE_OFFSET_BITS) || defined(_LARGEFILE_SOURCE)
#endif
#if defined(NITRO_NRT_LIB) && defined(NITRO_NRT_DLL)
    #error "Both NITRO_NRT_LIB and NITRO_NRT_DLL are #define'd'"
#endif
#if defined(NITRO_NRT_EXPORTS) && defined(NITRO_NRT_LIB)
    #error "Can't export from a LIB'"
#endif

// https://www.gnu.org/software/gnulib/manual/html_node/Exported-Symbols-of-Shared-Libraries.html
#if !defined(NITRO_NRT_library_export) && !defined(NITRO_NRT_library_import)
    #if defined(__GNUC__) // && HAVE_VISIBILITY 
        // https://www.gnu.org/software/gnulib/manual/html_node/Exported-Symbols-of-Shared-Libraries.html
        #define NITRO_NRT_library_export __attribute__((visibility("default")))

        // For GCC, there's no difference in consuming ("importing") an .a or .so
        #define NITRO_NRT_library_import /* no __declspec(dllimport) for GCC */

    #elif defined(_MSC_VER) // && (defined(_WINDLL) && !defined(_LIB))
        #define NITRO_NRT_library_export __declspec(dllexport)

        // Actually, it seems that the linker is able to figure this out from the .LIB,
        // so there doesn't seem to be a need for __declspec(dllimport).  Clients don't
        // need to #define NITRO_NITFCPP_DLL ... ?  Well, almost ... it looks
        // like __declspec(dllimport) is needed to get virtual "inline"s (e.g.,
        // destructors) correct.
        #define NITRO_NRT_library_import __declspec(dllimport)

    #else
        // https://stackoverflow.com/a/2164853/8877
        #define NITRO_NRT_library_export /* do nothing and hope for the best? */
        #define NITRO_NRT_library_import /* do nothing and hope for the best? */
        #pragma warning Unknown dynamic link import semantics.
    #endif
#endif

// The following ifdef block is the standard way of creating macros which make exporting
// from a DLL simpler. All files within this DLL are compiled with the NITRO_NRT_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see
// NITRO_NRT_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef NITRO_NRT_EXPORTS
    #define NITRO_NRT_API NITRO_NRT_library_export
#else
    // Either building a static library (no NITRO_NRT_EXPORTS) or
    // importing (not building) a shared library.

    // We need to know whether we're consuming (importing) a DLL or static LIB
    // The default is a static LIB as that's what existing code/builds expect.
    #ifdef NITRO_NRT_DLL
        // Actually, it seems that the linker is able to figure this out from the .LIB, so 
        // there doesn't seem to be a need for __declspec(dllimport).  Clients don't
        // need to #define NITRO_NRT_DLL ... ?  Well, almost ... it looks
        // like __declspec(dllimport) is needed to get virtual "inline"s (e.g., 
        // destructors) correct.
        #define NITRO_NRT_API NITRO_NRT_library_import
    #else
        #define NITRO_NRT_API /* "importing" a static LIB */
    #endif
#endif

