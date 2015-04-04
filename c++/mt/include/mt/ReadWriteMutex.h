/* =========================================================================
 * This file is part of mt-c++ 
 * =========================================================================
 * 
 * (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
 *
 * mt-c++ is free software; you can redistribute it and/or modify
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


#ifndef __MT_READ_WRITE_MUTEX_INTERFACE_H__
#define __MT_READ_WRITE_MUTEX_INTERFACE_H__


#if !defined(__APPLE_CC__)

#include <str/Dbg.h>
#include <mt/Mutex.h>
#include <mt/Semaphore.h>

namespace mt
{
    
/*!
 *  \class ReadWriteMutex
 *  \brief Locks resources exclusively during writes while allowing
 *  simultaneous reads
 *
 */
class ReadWriteMutex
{
    public:
    //!  Constructor
    ReadWriteMutex(int maxReaders) : mSem(maxReaders)
    {
        mMaxReaders = maxReaders;
        dbg_printf("Creating a read/write mutex\n");
    }

    //!  Destructor
    virtual ~ReadWriteMutex()
    {
        dbg_printf("Destroying a read/write mutex\n");
    }

    /*!
     *  Lock for reading (no writes allowed)
     */
    virtual void lockRead();

    /*!
     *  Unlock for reading (writes allowed)
     */
    virtual void unlockRead();

    /*!
     *  Lock for writing (no reads/other writes allowed)
     */
    virtual void lockWrite();

    /*!
     *  Unlock for writing (reads allowed)
     */
    virtual void unlockWrite();

protected:
    mt::Semaphore mSem;
    mt::Mutex mMutex;
    int mMaxReaders;
};
    
}

#endif // Not apple

#endif

