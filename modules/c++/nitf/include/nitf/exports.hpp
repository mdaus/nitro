#ifndef NITRO_nitf_exports_hpp_INCLUDED_
#define NITRO_nitf_exports_hpp_INCLUDED_
#pragma once

// The following ifdef block is the standard way of creating macros which make exporting
// from a DLL simpler. All files within this DLL are compiled with the NITRO_NITFCPP_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see
// NITRO_NITFCPP_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
// https://www.gnu.org/software/gnulib/manual/html_node/Exported-Symbols-of-Shared-Libraries.html
#if NITRO_NITFCPP_EXPORTS
    #if HAVE_VISIBILITY && defined(__GNUC__)
        #define NITRO_NITFCPP_API __attribute__((visibility("default")))
    #elif defined(_MSC_VER) && (defined(_WINDLL) && !defined(_LIB))
        #define NITRO_NITFCPP_API __declspec(dllexport)
    #else
        // https://stackoverflow.com/a/2164853/8877
        #define NITRO_NITFCPP_API /* do nothing and hope for the best? */
        #pragma warning Unknown dynamic link export semantics.
    #endif
#else
    // Either building a static library (no NITRO_NITFCPP_EXPORTS) or
    // importing (not building) a shared library.
    #if defined(_MSC_VER)
        // We need to know whether we're consuming (importing) a DLL or static LIB
        // The default is a static LIB as that's what existing code/builds expect.
        #if NITRO_NITFCPP_DLL
            #define NITRO_NITFCPP_API __declspec(dllimport)
        #else
            #define NITRO_NITFCPP_API /* "importing" a static LIB */
        #endif
    #else // #if defined(__GNUC__)
        // For GCC, there's no difference in consuming ("importing) a .a or .so
        #define NITRO_NITFCPP_API /* no __declspec(dllimport) for GCC */
        // https://stackoverflow.com/a/2164853/8877
        //#define NITRO_NITFCPP_API /* do nothing and hope for the best? */
        //#pragma warning Unknown dynamic link import semantics.
    #endif
#endif

#endif // NITRO_nitf_exports_hpp_INCLUDED_
