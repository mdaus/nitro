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
#ifndef NRT_Exports_h_INCLUDED_
#define NRT_Exports_h_INCLUDED_

// Need to specify how this code will be consumed, either NRT_LIB (static library)
// or NRT_DLL (aka "shared" library).  For DLLs, it needs to be set for BOTH
// "exporting" (building this code) and "importing" (consuming).
//
// Use Windows naming conventions (DLL, LIB) because this really only matters for _MSC_VER, see below.
#if !defined(NRT_LIB) && !defined(NRT_DLL)
    #define NRT_LIB 1
    //#define NRT_DLL 1
#endif
#if defined(NRT_LIB) && defined(NRT_DLL)
    #error "Both NRT_LIB and NRT_DLL are #define'd'"
#endif
#if defined(NRT_EXPORTS) && defined(NRT_LIB)
    #error "Can't export from a LIB'"
#endif

// https://www.gnu.org/software/gnulib/manual/html_node/Exported-Symbols-of-Shared-Libraries.html
#if !defined(NRT_library_export) && !defined(NRT_library_import)
    #if defined(__GNUC__) // && HAVE_VISIBILITY 
        // https://www.gnu.org/software/gnulib/manual/html_node/Exported-Symbols-of-Shared-Libraries.html
        #define NRT_library_export __attribute__((visibility("default")))

        // For GCC, there's no difference in consuming ("importing") an .a or .so
        #define NRT_library_import /* no __declspec(dllimport) for GCC */

    #elif defined(_MSC_VER) // && (defined(_WINDLL) && !defined(_LIB))
        #define NRT_library_export __declspec(dllexport)

        // Actually, it seems that the linker is able to figure this out from the .LIB,
        // so there doesn't seem to be a need for __declspec(dllimport).  Clients don't
        // need to #define NITRO_NITFCPP_DLL ... ?  Well, almost ... it looks
        // like __declspec(dllimport) is needed to get virtual "inline"s (e.g.,
        // destructors) correct.
        #define NRT_library_import __declspec(dllimport)

    #else
        // https://stackoverflow.com/a/2164853/8877
        #define NRT_library_export /* do nothing and hope for the best? */
        #define NRT_library_import /* do nothing and hope for the best? */
        #pragma warning Unknown dynamic link import semantics.
    #endif
#endif

// The following ifdef block is the standard way of creating macros which make exporting
// from a DLL simpler. All files within this DLL are compiled with the NRT_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see
// NRT_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef NRT_EXPORTS
    #define NRT_API NRT_library_export
#else
    // Either building a static library (no NRT_EXPORTS) or
    // importing (not building) a shared library.

    // We need to know whether we're consuming (importing) a DLL or static LIB
    // The default is a static LIB as that's what existing code/builds expect.
    #ifdef NRT_DLL
        // Actually, it seems that the linker is able to figure this out from the .LIB, so 
        // there doesn't seem to be a need for __declspec(dllimport).  Clients don't
        // need to #define NRT_DLL ... ?  Well, almost ... it looks
        // like __declspec(dllimport) is needed to get virtual "inline"s (e.g., 
        // destructors) correct.
        #define NRT_API NRT_library_import
    #else
        #define NRT_API /* "importing" a static LIB */
    #endif
#endif

#endif // NRT_Exports_h_INCLUDED_
