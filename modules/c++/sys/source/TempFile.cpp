/* =========================================================================
 * This file is part of sys-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2016, MDA Information Systems LLC
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
 * License along with this program; If not,
 * see <http://www.gnu.org/licenses/>.
 *
 */

#include <cstdio>
#include <string>
#include <fstream>
#include <import/except.h>
#include "sys/Conf.h"
#include "sys/TempFile.h"

sys::TempFile::TempFile(bool destroy) :
    mDestroy(destroy)
{
#if defined(WIN32)
    // MSDN states this may not be longer than MAX_PATH - 14,
    // or GetTempFileName() will fail
    static const size_t MAX_TEMP_PATH = MAX_PATH - 14;
    char pathBuffer[MAX_TEMP_PATH];
    size_t pathLength = GetTempPath(MAX_TEMP_PATH, pathBuffer);
    if (pathLength > 0 && pathLength <= MAX_TEMP_PATH)
    {
        // Error should only happen if pathBuffer is too long,
        // which we've already checked
        char tempFile[MAX_PATH];
        GetTempFileName(pathBuffer, "tmp", 0, tempFile);
        mName = tempFile;
    }
    else
    {
        throw except::Exception(Ctxt("An error occured while creating a "
                "temporary file."));
    }
#elif defined(__GNUC__)
    char nameBuffer[] = "/tmp/XXXXXX"; // mkstemp replaces the X's
    if (mkstemp(nameBuffer) != -1)
    {
        mName = nameBuffer;
    }
    else
    {
        throw except::Exception(Ctxt("An error occured while creating a "
                "temporary file."));
    }
#else
    // This method is cross-platform, but not secure
    // L_tmpnam is the longest name that std::tmpnam may return
    char nameBuffer[L_tmpnam];
    if (std::tmpnam(nameBuffer))
    {
        mName = nameBuffer;
        // Open and close file so it actually exists
        fclose(fopen(nameBuffer, "w"));
    }
    else
    {
        throw except::Exception(Ctxt("An error occured while calling tmpnam."
                " Unable to create temporary file"));
    }
#endif
}

sys::TempFile::~TempFile()
{
    if (mDestroy)
    {
        std::remove(mName.c_str());
    }
}

std::string sys::TempFile::pathname() const
{
    return mName;
}

