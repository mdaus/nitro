/* =========================================================================
 * This file is part of sys-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
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

#ifndef __SYS_ATOMIC_COUNTER_H__
#define __SYS_ATOMIC_COUNTER_H__

#include <assert.h>

#include <sys/Conf.h>
#include <sys/Mutex.h>

namespace sys
{
/*!
 *  \class AtomicUint
 *  \brief This class provides atomic incrementing, decrementing, and setting
 *         of an unsigned integer.  All operations are thread-safe.
 *
 *  TODO: There are other operations we could provide such as compareAndSet()
 *        if they're useful.
 *  TODO: The mutex locking/unlocking is way more heavy-duty than we need.
 *        There are platform-specific instructions for these sorts of
 *        operations.  For reference,
 *        boost/smart_ptr/detail/atomic_count_*.hpp probably implements this
 *        for every platform we need.
 */
class AtomicUint
{
public:
    typedef Uint64_T ValueType;

    //! Constructor
    AtomicUint(ValueType initialValue = 0) :
        mValue(initialValue)
    {
    }

    /*!
     *   Increment the value
     *   \return The value PRIOR to incrementing
     */
    ValueType getAndIncrement()
    {
        lock();
        const ValueType prevValue(mValue);
        ++mValue;
        unlock();

        return prevValue;
    }

    /*!
     *   Increment the value
     *   \return The value AFTER incrementing
     */
    ValueType incrementAndGet()
    {
        lock();
        ++mValue;
        const ValueType updatedValue(mValue);
        unlock();

        return updatedValue;
    }

    //! Increment the value
    void increment()
    {
        getAndIncrement();
    }

    /*!
     *   Decrement the value
     *   \return The value PRIOR to decrementing
     */
    ValueType getAndDecrement()
    {
        lock();
        const ValueType prevValue(mValue);
        --mValue;
        unlock();

        return prevValue;
    }

    /*!
     *   Decrement the value
     *   \return The value AFTER decrementing
     */
    ValueType decrementAndGet()
    {
        lock();
        --mValue;
        const ValueType updatedValue(mValue);
        unlock();

        return updatedValue;
    }

    //! Decrement the value
    void decrement()
    {
        getAndDecrement();
    }

    /*!
     *   Increment the value
     *   \return The value PRIOR to incrementing
     */
    ValueType getAndSet(ValueType value)
    {
        lock();
        const ValueType prevValue(mValue);
        mValue = value;
        unlock();

        return prevValue;
    }

    //! Set the value
    void set(ValueType value)
    {
        getAndSet(value);
    }

    /*!
     *   Get the current value
     *   \return The current value
     */
    ValueType get() const
    {
        lock();
        const ValueType value(mValue);
        unlock();

        return value;
    }

private:
    // Noncopyable
    AtomicUint(const AtomicUint& );
    const AtomicUint& operator=(const AtomicUint& );

    void lock() const
    {
        assert(mMutex.lock());
    }

    void unlock() const
    {
        assert(mMutex.unlock());
    }

private:
    mutable Mutex mMutex;
    ValueType     mValue;
};
}

#endif
