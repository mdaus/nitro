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


#ifndef __MT_MUTEX_INTERFACE_H__
#define __MT_MUTEX_INTERFACE_H__


#include <typeinfo>
#include <mt/MultiThreadException.h>
#include <str/Dbg.h>

namespace mt
{

/*!
 *  \class MutexInterface
 *  \brief The interface for any mutex
 *
 *  This class defines the interface for any mutex in any package that
 *  is wrapped herein
 */
class MutexInterface
{
public:
    //!  Constructor
    MutexInterface()
    {
#ifdef THREAD_DEBUG
        dbg_printf("Creating a mutex\n");
#endif

    }

    //!  Destructor
    virtual ~MutexInterface()
    {
#ifdef THREAD_DEBUG
        dbg_printf("Destroying a mutex\n");
#endif

    }

    /*!
     *  Lock the mutex up.
     */
    virtual void lock() = 0;

    /*!
     *  Unlock the mutex.
     */
    virtual void unlock() = 0;

};

}

#endif
