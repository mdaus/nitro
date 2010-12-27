/* =========================================================================
 * This file is part of logging-c++ 
 * =========================================================================
 * 
 * (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
 *
 * logging-c++ is free software; you can redistribute it and/or modify
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

///////////////////////////////////////////////////////////
//  RotatingFileHandler.cpp
///////////////////////////////////////////////////////////

#include "logging/RotatingFileHandler.h"

using namespace logging;

RotatingFileHandler::RotatingFileHandler(const std::string& fname,
                                         long maxBytes, int backupCount,
                                         LogLevel level) :
    StreamHandler(level)
{
    sys::OS os;
    int creationFlags;

    if (!os.exists(fname))
    {
        //see if we need to make the parent directory
        std::string parDir = sys::Path::splitPath(fname).first;
        if (!os.exists(parDir))
            os.makeDirectory(parDir);
        creationFlags = sys::File::CREATE | sys::File::TRUNCATE;
    }
    else
    {
        creationFlags = sys::File::EXISTING;
    }
    mStream.reset(new io::RotatingFileOutputStream(fname, maxBytes,
                                                   backupCount, creationFlags));
}

RotatingFileHandler::~RotatingFileHandler()
{
    // the StreamHandler destructor closes the stream
}
