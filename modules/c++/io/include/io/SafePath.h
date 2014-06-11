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

#ifndef __IO_SAFEPATH_H__
#define __IO_SAFEPATH_H__

#include <string>

namespace io
{

/*!
 * \brief SafePath class
 *
 * Class to protect against writing a broken file which
 * appears to be complete
 */
class SafePath
{
public:

    /*!
     * Create a SafePath object and generate a unique, temporary name
     */
    SafePath(const std::string& realPathname);

    /*!
     * Destruct SafePath object
     * This either changes the name of the file back to realName or deletes
     * the temp file depending on whether or not the moved flag has been set
     */
    ~SafePath();

    /*!
     * Return the temporary filename
     * Throws if file has already been moved
     */
    std::string getTempPathname() const;

    /*!
     * Tell the object that file processing was successful
     * and that it should rename the file.
     */
    void moveFile();

private:

    const std::string mRealPathname;
    const std::string mTempPathname;
    bool mMoved;
};

}

#endif /* __IO_SAFEPATH_H__ */
