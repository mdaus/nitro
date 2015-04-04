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


#if defined(USE_NSPR_THREADS)
#include "mt/ConditionVarNSPR.h"

mt::ConditionVarNSPR::ConditionVarNSPR() :
    mMutexOwned(new mt::MutexNSPR()),
    mMutex(mMutexOwned.get())
{
    mNative = PR_NewCondVar( (mMutex->getNative()) );
    if (mNative == NULL)
        throw mt::ThreadResourceException("Condition Variable initialization failed");
}

mt::CondtionVarNSPR::ConditionVarNSPR(mt::MutexNSPR *theLock, bool isOwner) :
    mMutex(theLock)
{
    if (!theLock)
        throw except::NullPointerReference("ConditionVar received NULL mutex");

    if (isOwner)
        mMutexOwned.reset(theLock);

    mNative = PR_NewCondVar( (mMutex->getNative()) );
    if (mNative == NULL)
        throw mt::ThreadResourceException("Condition Variable initialization failed");
}

mt::ConditionVarNSPR::~ConditionVarNSPR()
{
    PR_DestroyCondVar(mNative);
}

void mt::ConditionVarNSPR::acquireLock()
{
    mMutex->lock();
}

void mt::ConditionVarNSPR::dropLock()
{
    mMutex->unlock();
}

void mt::ConditionVarNSPR::signal()
{
    if (PR_NotifyCondVar(mNative) != PR_SUCCESS)
        throw mt::ConditionVarException("Condition Variable signal failed");

}

void mt::ConditionVarNSPR::wait()
{
    if (PR_WaitCondVar(mNative, PR_INTERVAL_NO_WAIT) != PR_SUCCESS)
        throw mt::ConditionVarException("Condition Variable wait failed");
}

void mt::ConditionVarNSPR::wait(double seconds)
{
    double milli = seconds * 1000;
    if (PR_WaitCondVar(mNative, PR_MillisecondsToInterval((PRUint32) milli)) != PR_SUCCESS)
        throw mt::ConditionVarException("Condition Variable wait failed");
}

void mt::ConditionVarNSPR::broadcast()
{
    if (PR_NotifyAllCondVar(mNative) != PR_SUCCESS)
        throw mt::ConditionVarException("Condition Variable broadcast failed");
}

PRCondVar*& mt::ConditionVarNSPR::getNative()
{
    return mNative;
}

#endif

