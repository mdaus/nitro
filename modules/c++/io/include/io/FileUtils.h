/* =========================================================================
 * This file is part of io-c++ 
 * =========================================================================
 * 
 * (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
 *
 * io-c++ is free software; you can redistribute it and/or modify
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

#ifndef __IO_FILE_UTILS_H__
#define __IO_FILE_UTILS_H__

#include <import/except.h>
#include <import/str.h>
#include <import/sys.h>

namespace io
{

/*!
 *  Copy a file or directory to a new path. 
 *  Source and destination cannot be the same location
 *
 *  \param path      - source location
 *  \param newath    - destination location
 *  \param blockSize - files are copied in blocks (1MB default)
 *  \param recurse   - recursively copy
 *  \return True upon success, false if failure
 */
bool copy(const std::string& path, 
          const std::string& newPath,
          size_t blockSize = 1048576,
          bool recurse = true);

/*!
 *  Move file with this path name to the newPath
 *  \return True upon success, false if failure
 */
inline bool move(const std::string& path, 
                 const std::string& newPath)
{
    sys::OS os;
    os.move(path, newPath);
}

/*!
 *  Remove file with this path name
 *  \return True upon success, false if failure
 */
inline bool remove(const std::string& path, bool recursive = true)
{
    sys::OS os;
    os.remove(path, recursive);
}


/**
 * Static file manipulation utilities.
 */
class FileUtils
{
public:
    /**
     * Creates a file in the given directory. It will try to use the filename
     * given. If overwrite is false, it will create a filename based on the
     * given one. If the given filename is empty, a temporary name is used.
     */
    static std::string createFile(std::string dirname, std::string filename =
            std::string(""), bool overwrite = true) throw (except::IOException);

    static void touchFile(std::string filename) throw (except::IOException);

    static void forceMkdir(std::string dirname) throw (except::IOException);

private:
    //private constructor
    FileUtils()
    {
    }
};

}

#endif
