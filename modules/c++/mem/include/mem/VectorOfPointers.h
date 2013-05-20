/* =========================================================================
 * This file is part of mem-c++
 * =========================================================================
 *
 * (C) Copyright 2013, General Dynamics - Advanced Information Systems
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

#ifndef __MEM_VECTOR_OF_POINTERS_H__
#define __MEM_VECTOR_OF_POINTERS_H__

#include <cstddef>
#include <vector>

namespace mem
{
    /*!
     *  \class VectorOfPointers
     *  \brief This class provides safe cleanup for vectors of pointers
     */
    template <typename T>
    class VectorOfPointers
    {
    public:
        VectorOfPointers()
        {
        }

        ~VectorOfPointers()
        {
            clear();
        }

        void clear()
        {
            for(size_t i=0; i < mVec.size(); ++i)
                if(mVec[i])
                    delete mVec[i];
            mVec.clear();
        }

        std::vector<T*>& get() const
        {
            return mVec;
        }

        T* operator[](std::ptrdiff_t idx) const
        {
            return mVec[idx];
        }

        void push_back(T* elem)
        {
            mVec.push_back(elem);
        }

        T* back()
        {
            return mVec.back();
        }

        const T* back() const
        {
            return mVec.back();
        }

    private:
        // Noncopyable
        VectorOfPointers(const VectorOfPointers& );
        const VectorOfPointers& operator=(const VectorOfPointers& );

    private:
        std::vector<T*> mVec;
    };
}

#endif
