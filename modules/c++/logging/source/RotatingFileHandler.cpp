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


void logging::RotatingFileHandler::emitRecord(logging::LogRecord* record)
{
    if (shouldRollover(record))
        doRollover();
    logging::FileHandler::emitRecord(record);
}


bool logging::RotatingFileHandler::shouldRollover(LogRecord* record)
{
    if (mMaxBytes > 0)
    {
        std::string msg = format(record);
        io::FileOutputStream* fos = (io::FileOutputStream*)mStream.get();
        sys::Off_T pos = fos->tell();

        if (pos + msg.length() > mMaxBytes)
            return true;
    }
    return false;
}


void logging::RotatingFileHandler::doRollover()
{
    io::FileOutputStream* fos = (io::FileOutputStream*)mStream.get();
    fos->close();
    sys::OS os;

    if (mBackupCount > 0)
    {
        for (int i = mBackupCount - 1; i > 0; --i)
        {
            std::stringstream curName;
            curName << mName << "." << i;
            std::stringstream nextName;
            nextName << mName << "." << (i + 1);
            if (os.exists(curName.str()))
            {
                if (os.exists(nextName.str()))
                {
                    os.remove(nextName.str());
                }
                os.move(curName.str(), nextName.str());
            }
        }
        std::string curName = mName + ".1";
        if (os.exists(curName))
            os.remove(curName);
        os.move(mName, curName);
    }

    //reopen the "new" base filename
#ifdef USE_IO_STREAMS
    fos->open(mName.c_str(), std::ios::out | std::ios::app);
#else
    fos->create(mName.c_str());
#endif
}

