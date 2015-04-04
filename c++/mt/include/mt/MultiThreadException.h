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


#ifndef __MULTI_THREAD_EXCEPTION_H__
#define __MULTI_THREAD_EXCEPTION_H__

#include <except/Error.h>
#include <except/Exception.h>

/*!
 *  \file  MultiThreadException.h
 *  \brief Provide exceptions that are specialized for multithreading contexts
 */
namespace mt
{
/*!
 *  \class ConditionVarException
 *  \brief An exception class for condition variables
 */
DECLARE_EXCEPTION(ConditionVar)
 
/*!
 *  \class ThreadResourceException
 *  \brief An exception class for thread resources
 */
DECLARE_EXCEPTION(ThreadResource)

/*!
 *  \class LockException
 *  \brief An exception class for locking and unlocking
 */
DECLARE_EXCEPTION(Lock)
 
}
 
 #endif