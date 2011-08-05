/* =========================================================================
 * This file is part of mem-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
 *
 * mem-c++ is free software; you can redistribute it and/or modify
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

#ifndef __MEM_SHARED_PTR_H__
#define __MEM_SHARED_PTR_H__

#include <memory>

#include <sys/AtomicUint.h>

namespace mem
{
/*!
 *  \class SharedPtr
 *  \brief This class provides RAII for object allocations via new.
 *         Additionally, it uses thread-safe reference counting so that the
 *         underlying pointer can be shared among multiple objects.  When the
 *         last SharedPtr goes out of scope, the underlying pointer is
 *         deleted.
 */
template <class T>
class SharedPtr
{
public:
    explicit SharedPtr(T* ptr = NULL) :
        mPtr(ptr)
    {
        // NOTE: The only time mRefCtr is NULL is when the underlying pointer
        //       is NULL.
        // TODO: This is done to make default construction cheaper since the
        //       AtomicUint implementation currently uses an expensive mutex.
        //       Once this is changed to use atomic operations, this logic
        //       can be removed.

        if (ptr)
        {
            // Initially we have a reference count of 1
            // In the constructor, we take ownership of the pointer no matter
            // what, so if we're having a really bad day and creating the
            // reference counter throws, we need to delete the input pointer
            std::auto_ptr<T> scopedPtr(ptr);
            mRefCtr = new sys::AtomicUint(1);
            scopedPtr.release();
        }
        else
        {
            mRefCtr = NULL;
        }
    }

    explicit SharedPtr(std::auto_ptr<T> ptr) :
        mPtr(ptr.get())
    {
        if (mPtr)
        {
            // Initially we have a reference count of 1
            // If this throws, the auto_ptr will clean up the input pointer
            // for us
            mRefCtr = new sys::AtomicUint(1);

            // We now own the pointer
            ptr.release();
        }
        else
        {
            mRefCtr = NULL;
        }
    }

    ~SharedPtr()
    {
        if (mRefCtr && mRefCtr->decrementAndGet() == 0)
        {
            delete mRefCtr;
            delete mPtr;
        }
    }

    SharedPtr(const SharedPtr& rhs) :
        mRefCtr(rhs.mRefCtr),
        mPtr(rhs.mPtr)
    {
        if (mRefCtr)
        {
            mRefCtr->increment();
        }
    }

    const SharedPtr&
    operator=(const SharedPtr& rhs)
    {
        if (this != &rhs)
        {
            if (mRefCtr && mRefCtr->decrementAndGet() == 0)
            {
                // We were holding the last copy of this data prior to this
                // assignment - need to clean it up
                delete mRefCtr;
                delete mPtr;
            }

            mRefCtr = rhs.mRefCtr;
            mPtr = rhs.mPtr;
            if (mRefCtr)
            {
                mRefCtr->increment();
            }
        }

        return *this;
    }

    T* get() const
    {
        return mPtr;
    }

    T& operator*() const
    {
        return *mPtr;
    }

    T* operator->() const
    {
        return mPtr.operator->();
    }

    sys::AtomicUint::ValueType getCount() const
    {
        return (mRefCtr ? mRefCtr->get() : 0);
    }

    void reset(T* ptr = NULL)
    {
        // We've agreed to take ownership of the pointer no matter what, so
        // if we're having a really bad day and creating the reference counter
        // throws, we need to delete the input pointer
        // NOTE: We need to do this on the side before decrementing mRefCtr.
        //       This way, we can provide the strong exception guarantee (i.e.
        //       the operation either succeeds or throws - the underlying
        //       object is always in a good state).
        sys::AtomicUint* newRefCtr = NULL;
        if (ptr)
        {
            std::auto_ptr<T> scopedPtr(ptr);
            newRefCtr = new sys::AtomicUint(1);
            scopedPtr.release();
        }

        if (mRefCtr && mRefCtr->decrementAndGet() == 0)
        {
            // We were holding the last copy of this data prior to this
            // reset - need to clean up
            delete mRefCtr;
            delete mPtr;
        }

        mRefCtr = newRefCtr;
        mPtr = ptr;
    }

private:
    sys::AtomicUint* mRefCtr;
    T*               mPtr;
};
}

#endif
