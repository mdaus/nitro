/* =========================================================================
 * This file is part of lang-c++
 * =========================================================================
 *
 * (C) Copyright 2004 - 2010, General Dynamics - Advanced Information Systems
 *
 * lang-c++ is free software; you can redistribute it and/or modify
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

#ifndef __LANG_HASH_MAP_H__
#define __LANG_HASH_MAP_H__

#include "lang/Map.h"
#include "lang/STLList.h"
#include "lang/STLVector.h"
#include "lang/Defaults.h"
#include <map>

namespace lang    
{

/**
 * Compares the keys of Hash Pairs
 */
template <typename Pair_T, typename KeyComparator_T>
class HashPairComparator
{
public:
    ~HashPairComparator(){}
    virtual bool operator() (const Pair_T& obj1, const Pair_T& obj2) const
    {
        return mComparator(obj1.first, obj2.first);
    }
protected:
    KeyComparator_T mComparator;
};

/**
 * Clones a Hash Pair
 */
template <typename Key_T, typename Value_T, typename ValueCloner_T>
class HashPairCloner
{
public:
    typedef ::lang::Pair<Key_T, Value_T> Pair;
    ~HashPairCloner(){}
    virtual Pair operator() (const Pair& obj) const
    {
        return Pair(obj.first, mCloner(obj.second));
    }
protected:
    ValueCloner_T mCloner;
};

/**
 * Deletes the Value of the pair
 */
template <typename Pair_T, typename ValueDestructor_T>
class HashPairDestructor
{
public:
    ~HashPairDestructor(){}
    virtual void operator() (Pair_T& obj)
    {
        mDestructor(obj.second);
    }
protected:
    ValueDestructor_T mDestructor;
};

/**
 * Iterator for HashMap
 */
template<typename K, typename V, typename LLComparator_T,
         typename LLCloner_T, typename LLPairDestructor_T,
         typename ALComparator_T, typename ALCloner_T,
         typename ALPairDestructor_T>
class HashMapIterator: public Iterator<Pair<K, V> >
{
public:
    typedef STLList<Pair<K, V>, LLComparator_T,
                    LLCloner_T, LLPairDestructor_T> Bucket_T;
    typedef STLVector<Bucket_T*, ALComparator_T, ALCloner_T,
                      ALPairDestructor_T> Buckets_T;

    HashMapIterator(Buckets_T *buckets) :
        mBuckets(buckets), mBucket(NULL)
    {
        mBucketsIterator = mBuckets->iterator();
        seekToNext();
    }

    virtual ~HashMapIterator()
    {
    }

    bool hasNext() const
    {
        return mBucket && mBucketIterator->hasNext();
    }

    Pair<K, V> next()
    {
        if (!hasNext())
            throw except::NullPointerReference(Ctxt("No elements left"));
        Pair<K, V> p = mBucketIterator->next();
        if (!mBucketIterator->hasNext())
        {
            seekToNext();
        }
        return p;
    }

protected:
    Buckets_T *mBuckets;
    Bucket_T *mBucket;
    typedef typename Buckets_T::Iterator BucketsIterator;
    typedef typename Bucket_T::Iterator BucketIterator;

    BucketsIterator mBucketsIterator;
    BucketIterator mBucketIterator;

    void seekToNext()
    {
        while(mBucketsIterator->hasNext())
        {
            mBucket = mBucketsIterator->next();
            if (mBucket)
            {
                mBucketIterator = mBucket->iterator();
                if(mBucketIterator->hasNext())
                    break;
            }
        }
    }
};


/**
 * Hash map implementation of Map
 */
template<typename Key_T,
         typename Value_T,
         typename Hasher_T = DefaultStringHash<Key_T>,
         typename KeyComparator_T = DefaultComparator<Key_T>,
         typename ValueCloner_T = DefaultCloner<Value_T>,
         typename ValueDestructor_T = DefaultDestructor<Value_T> >
class HashMap: public Map<Key_T, Value_T>
{
public:
    const static size_t DEFAULT_BUCKETS = 50;
    typedef ::lang::Pair<Key_T, Value_T> Pair;
    typedef HashMap<Key_T, Value_T, Hasher_T, KeyComparator_T,
                    ValueCloner_T, ValueDestructor_T> Map_T;
    typedef std::auto_ptr< ::lang::Iterator<Pair> > Iterator;

    HashMap(size_t buckets = DEFAULT_BUCKETS) : mSize(0)
    {
        mBuckets = new Buckets_T(buckets);
        for(size_t i = 0; i < buckets; ++i)
            mBuckets->insert(NULL);
    }
    virtual ~HashMap()
    {
        destroy(true);
    }

    virtual Iterator iterator() const
    {
        return Iterator(new HashMapIterator<Key_T, Value_T, BucketComparator_T, BucketCloner_T,
                               BucketDestructor_T, BucketsComparator_T,
                               BucketsCloner_T, BucketsDestructor_T>(mBuckets));
    }

    virtual bool exists(Key_T& key) const
    {
        try
        {
            findValue(key);
            return true;
        }
        catch(except::NoSuchKeyException&)
        {
            return false;
        }
    }

    virtual size_t size() const
    {
        size_t num = 0;
        BucketsIterator it = mBuckets->iterator();
        while(it->hasNext())
        {
            Bucket_T *bucket = it->next();
            if (bucket)
                num += bucket->size();
        }
        return num;
    }

    virtual Value_T operator[](const Key_T& key) const
            throw (except::NoSuchKeyException)
    {
        return findValue(key);
    }

    virtual Value_T pop(const Key_T& key) throw (except::NoSuchKeyException)
    {
        size_t index = hashIt(key);
        Bucket_T *bucket = NULL;
        try
        {
            bucket = mBuckets->get(index);
        }
        catch(except::IndexOutOfRangeException& ex)
        {
        }
        if (bucket)
        {
            //see if it is already there
            Iterator it = bucket->iterator();
            size_t index = 0;
            while(it->hasNext())
            {
                Pair p = it->next();
                if (mComparator(p.first, key))
                {
                    bucket->remove(p);
                    return p.second;
                }
            }
        }
        throw except::NoSuchKeyException(Ctxt(str::toString(key)));
    }

    virtual bool remove(const Key_T& key)
    {
        try
        {
            Value_T v = pop(key);
            mDestructor(v);
            return true;
        }
        catch(except::NoSuchKeyException& ex)
        {
            return false;
        }
    }

    virtual Value_T& operator[](const Key_T& key)
    {
        size_t index = hashIt(key);
        Bucket_T *bucket = NULL;
        try
        {
            bucket = mBuckets->get(index);
        }
        catch(except::IndexOutOfRangeException& ex)
        {
        }
        if (!bucket)
        {
            (*mBuckets)[index] = bucket = new Bucket_T;
        }
        else
        {
            //see if it is already there
            Iterator it = bucket->iterator();
            size_t index = 0;
            while(it->hasNext())
            {
                Pair p = it->next();
                if (mComparator(p.first, key))
                    return (*bucket)[index].second;
                ++index;
            }
        }
        //It's not here yet, so we add a new one
        Pair p;
        p.first = key;
        bucket->append(p);
        return bucket->back().second;
    }

    virtual void clear()
    {
        destroy(false);
    }

    virtual Map_T* clone() const
    {
        TrueFilter<Pair> identity;
        return filter(identity);
    }

    virtual Map_T* filter(Filter<Pair>& filter) const
    {
        Map_T *filtered = new Map_T(mBuckets->capacity());
        Iterator it = iterator();
        while (it->hasNext())
        {
            Pair p = it->next();
            if (filter(p))
            {
                filtered->put(p.first, mCloner(p.second));
            }
        }
        return filtered;
    }

protected:
    typedef HashPairComparator<Pair, KeyComparator_T> BucketComparator_T;
    typedef HashPairCloner<Key_T, Value_T, ValueCloner_T> BucketCloner_T;
    typedef HashPairDestructor<Pair, ValueDestructor_T> BucketDestructor_T;
    typedef STLList<Pair, BucketComparator_T, BucketCloner_T, BucketDestructor_T> Bucket_T;
    typedef DefaultComparator<Bucket_T*> BucketsComparator_T;
    typedef PointerCloner<Bucket_T*> BucketsCloner_T;
    typedef DeleteDestructor<Bucket_T*> BucketsDestructor_T;
    typedef STLVector<Bucket_T*, BucketsComparator_T, BucketsCloner_T, BucketsDestructor_T> Buckets_T;
    typedef typename Buckets_T::Iterator BucketsIterator;

    Buckets_T* mBuckets;
    size_t mSize;
    Hasher_T mHasher;
    KeyComparator_T mComparator;
    ValueCloner_T mCloner;
    ValueDestructor_T mDestructor;

    virtual Value_T findValue(const Key_T& key) const
    {
        size_t index = hashIt(key);
        const Bucket_T *bucket = mBuckets->get(index);
        if (bucket)
        {
            Iterator it = bucket->iterator();
            while(it->hasNext())
            {
                Pair p = it->next();
                if (mComparator(p.first, key))
                    return p.second;
            }
        }
        throw except::NoSuchKeyException(Ctxt(str::toString(key)));
    }

    size_t hashIt(const Key_T& key) const
    {
        return mHasher(key) % mBuckets->capacity();
    }

    void destroy(bool destroyBuckets)
    {
        if (mBuckets)
        {
            BucketsIterator it = mBuckets->iterator();
            if (destroyBuckets)
                delete mBuckets;
            else
                mBuckets->clear();
        }
    }
};

}

#endif
