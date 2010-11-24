/* =========================================================================
 * This file is part of cli-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2010, General Dynamics - Advanced Information Systems
 *
 * cli-c++ is free software; you can redistribute it and/or modify
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

#ifndef __CLI_VALUE_H__
#define __CLI_VALUE_H__

#include <import/sys.h>
#include <import/str.h>

namespace cli
{

/**
 * The Value class provides access to one or more actual values. It provides index-based access to parameters.
 */
class Value
{
public:
    Value() : mStorage(new ArrayStorage)
    {
    }

    template <typename T>
    explicit Value(std::vector<T> value) : mStorage(NULL)
    {
        setContainer<T>(value);
    }

    template <typename T>
    Value(T value) : mStorage(NULL)
    {
        set<T>(value);
    }

    template <typename T>
    Value(T* value, size_t size, bool own = false) : mStorage(NULL)
    {
        set<T>(value, size, own);
    }

    ~Value() { cleanup(); }

    template <typename T>
    void set(T value)
    {
        cleanup();
        mStorage = new ScalarStorage(str::toString(value));
    }

    template <typename T>
    void set(T* value, size_t size, bool own = false)
    {
        cleanup();
        std::vector<std::string> vec(size);
        for(size_t i = 0; i < size; ++i)
            vec[i] = str::toString(value[i]);
        mStorage = new ArrayStorage(vec);
        if (own)
            delete [] value;
    }

    template <typename T>
    void setContainer(const std::vector<T>& c)
    {
        ArrayStorage *as = new ArrayStorage;
        std::copy(c.begin(), c.end(), std::back_inserter(as->value));
        mStorage = as;
    }

    template <typename T>
    T operator [] (unsigned int index) const
    {
        return at<T>(index);
    }

    template <typename T>
    T at(unsigned int index = 0) const
    {
        switch(mStorage->type)
        {
        case STORAGE_SCALAR:
            return str::toType<T>(((ScalarStorage*)mStorage)->value);
        case STORAGE_ARRAY:
            ArrayStorage* a = (ArrayStorage*)mStorage;
            if (index >= a->value.size())
                throw except::IndexOutOfRangeException(
                        Ctxt(FmtX("Invalid index: %d", index)));
            return str::toType<T>(a->value[index]);
        }
        throw except::Exception(Ctxt("Unsupported storage type"));
    }

    template <typename T>
    T get(unsigned int index = 0) const
    {
        return at<T>(index);
    }

    template <typename T>
    void add(T val)
    {
        if (mStorage->type == STORAGE_SCALAR)
        {
            ArrayStorage *a = new ArrayStorage;
            a->value.push_back(((ScalarStorage*)mStorage)->value);
            cleanup();
            mStorage = a;
        }
        if (mStorage->type == STORAGE_ARRAY)
        {
            ArrayStorage* a = (ArrayStorage*)mStorage;
            a->value.push_back(str::toString(val));
        }
        else
            throw except::Exception(Ctxt("Unsupported storage type"));
    }

    /**
     * Returns the size of value. Scalars always have a size of 1. Arrays return the number of elements of the
     */
    unsigned int size() const
    {
        switch(mStorage->type)
        {
        case STORAGE_SCALAR:
            return 1;
        case STORAGE_ARRAY:
            return ((ArrayStorage*)mStorage)->value.size();
        }
        throw except::Exception(Ctxt("Unsupported storage type"));
    }

    Value* clone() const
    {
        switch(mStorage->type)
        {
        case STORAGE_SCALAR:
            return new Value(((ScalarStorage*)mStorage)->value);
        case STORAGE_ARRAY:
            ArrayStorage* a = (ArrayStorage*)mStorage;
            return new Value(a->value);
        }
        throw except::Exception(Ctxt("Unsupported storage type"));
    }

    std::string toString() const
    {
        switch(mStorage->type)
        {
        case STORAGE_SCALAR:
            return ((ScalarStorage*)mStorage)->value;
        case STORAGE_ARRAY:
            ArrayStorage* a = (ArrayStorage*)mStorage;
            std::ostringstream s;
            s << "[" << str::join(a->value, ", ") << "]";
            return s.str();
        }
        throw except::Exception(Ctxt("Unsupported storage type"));
    }

protected:

    enum StorageType
    {
        STORAGE_SCALAR,
        STORAGE_ARRAY
    };

    struct Storage
    {
        Storage(StorageType t) : type(t){}
        StorageType type;
    };

    struct ScalarStorage : public Storage
    {
        ScalarStorage(std::string v) : Storage(STORAGE_SCALAR), value(v){}
        std::string value;
    };

    struct ArrayStorage : public Storage
    {
        ArrayStorage() : Storage(STORAGE_ARRAY) {}
        ArrayStorage(std::vector<std::string> arr) :
            Storage(STORAGE_ARRAY), value(arr){}
        std::vector<std::string> value;
    };

    Storage *mStorage;

    void cleanup() { if (mStorage) delete mStorage; }
};

}
#endif
