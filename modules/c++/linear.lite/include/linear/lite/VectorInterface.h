/* =========================================================================
 * This file is part of linear.lite-c++ 
 * =========================================================================
 * 
 * (C) Copyright 2004 - 2009, General Dynamics - Advanced Information Systems
 *
 * linear.lite-c++ is free software; you can redistribute it and/or modify
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

#ifndef __VECTOR_INTERFACE_H__
#define __VECTOR_INTERFACE_H__

#if defined(__sgi) || defined(__sgi__)
#  include <assert.h>
#  include <math.h>
#else
#  include <cassert>
#  include <cmath>
#endif

#include <import/sys.h>
#include <import/except.h>
#include "linear/lite/Cast.h"

/*!
 *  \file VectorInterface.h
 *  \brief Contains the interface for implementing vectors in this module
 *
 */

namespace linear
{
	namespace lite
	{
	/*!
	 *  \class VectorInterface
	 *  \brief Interface that math library drivers must implement
	 *
	 *  This class provides the interface for implementing a module compatible
	 *  vector.  The class is not intended to be used directly (though nothing prevents this).
	 *  Rather, users will instantiate using the VectorCreatorInterface implementations for
	 *  a specific driver, and will refer to the interface using the handle class Vector.
	 *
	 *  This class is not really a pure interface (although it contains no data structures).  
	 *  Several basic functions which may not have implementations in some libraries are
	 *  provided at this level.  Most libraries, however, will override these with more
	 *  streamlined implementations
	 *
	 */
	template<typename T> class VectorInterface
	{
	public:
		//!  Default constructor
		VectorInterface() {}

		/*!
		 *  Create a vector of specified size
		 *  \param n The size
		 */
		VectorInterface(int n) {}
		
		//!  Destructor
		virtual ~VectorInterface() {}

		/*!
		 *  Create a vector of specified size
		 *  \param n The size
		 */
		virtual void init(int n) = 0;
		
		/*!
		 *  Retrieve the size of the vector
		 *  \return The size
		 */
		virtual int size() const = 0;

		/*!
		 *  Clone the vector.
		 *  \return The new cloned vector
		 */
		virtual VectorInterface* clone() const = 0;
		
		/*!
		 *  Vector cross product.  Derived implementations currently only
		 *  support this call for 3 dimensional vectors.  For anything else
		 *  behavior is undefined (most implementations will throw in the future).
		 *
		 *  \param vec The vector to cross this by
		 *  \return The vector cross product
		 */
		virtual VectorInterface* cross(const VectorInterface& vec) const = 0;
	
		/*!
		 *  Add a vector to this and produce a new vector which is the sum
		 *  
		 *  \param vec A vector to add to this
		 *  \return A new vector sum
		 */
		virtual VectorInterface* add(const VectorInterface& vec) const = 0;
	
		/*!
		 *  Scale this vector.  This class provides a basic
		 *  implementation
		 *  
		 *  \param scalar amount to scale each element by
		 *
		 */
		virtual void scale(T scalar) 
		{
			for (int i = 0; i < size(); i++)
			{
				set(i, (get(i) * scalar));
			}
		}
		
		/*!
		 *  Dot this vector by another.  Return the result.
		 *  This class provides a basic implementation.
		 *
		 *  \param vec The vector to dot this by
		 *  \return The dot product
		 */
		virtual T dot(const VectorInterface& vec) const
		{
			assert(vec.size() == this->size());
			T acc(0);
			for (int i = 0; i < vec.size(); ++i)
			{
				acc += (get(i) * vec[i]);
			}
			return acc;
		}
		/*!
		 *  Euclidean (L2) Norm.  This class provides a basic implementation
		 *
		 *  \return The norm 2
		 */
		virtual T norm2() const
		{
			T acc(0);
			for (int i = 0; i < size(); ++i)
			{
				acc += (get(i) * get(i));
			}
			return (T)::sqrt((const T)acc);
		}

                /*!
                 *  Scale the vector by 1.0/norm
                 */
		virtual void normalize()
		{
                        this->scale(1.0/this->norm2());
		}

		/*!
		 *  Get the ith element in the vector (const)
		 *  \param i The index
		 *  \return The element (a copy)
		 */
		virtual T get(int i) const = 0;

		/*
		 *  Get the ith element in the vector
		 *  param i The index
		 *  return The element (a reference)
		 */
		virtual T& get(int i) 
                {
                    throw 
                        except::Exception(Ctxt("Not implemented, use set()!"));

                }

                /*!
                 *  Copy the data from the vector (const) to the given pointer
                 *  The pointer must be pre-allocated
                 *  \param dataPtr The pointer to copy the vector's data to
                 */
                void get(T* dataPtr) const
                {
                   int numElems = size();

                   for(int i=0; i<numElems; i++)
                   {
                      dataPtr[i] = get(i);
                   }
                }

		virtual void set(int i, const T& val) = 0;


                /*!
                 *  Copy the data from the given pointer to the vector
                 *  \param numElems The number of data elements to copy 
                 *  \param dataPtr The pointer to the data to copy 
                 */
                virtual void set(int numElems, const T* dataPtr)
                {
                   //check the size
                   if(size() != numElems);
                   {
                      init(numElems);
                   }

                   for (int i = 0; i < numElems; i++)
                   {
                      set(i, dataPtr[i]);
                   }
                }

		/*!
		 *  Get the ith element in the vector (const)
		 *  \param i The index
		 *  \return The element (a copy)
		 */
		T operator[](int i) const { return get(i); }

		/*
		 *  Get the ith element in the vector
		 *  param i The index
		 *  return The element (a reference)
		 */
		T& operator[](int i) { return get(i); }

		/*!
		 *  Print function for internal vector
		 */
		virtual void print() const = 0;
	};

	/*!
	 *  \class VectorCreatorInterface
	 *  \brief Interface for creating a vector of any underlying driver
	 *
	 *  This class is a pure interface for vector creation.  Drivers implement
	 *  this interface to produce implementations of vectors using an external library
	 */
	template<typename T> class VectorCreatorInterface
	{
	public:
		VectorCreatorInterface() {}
		virtual ~VectorCreatorInterface() {}
		virtual VectorInterface<T>* newVector(int d) const = 0;
		virtual VectorInterface<T>* newVector(int d, const T* dataPtr) const = 0;
		virtual VectorInterface<T>* newConstantVector(int d, T constantValue) const = 0;
	};
	
	/*!
	 *  \class VectorCreatorPartial
	 *  \brief Partial implementation for creator interface
	 *
	 *  This class provides a non-optimal implementation of the  newConstantVector method
	 *
	 */
	template<typename T> class VectorCreatorPartial : public VectorCreatorInterface<T>
	{

	public:
		//!  Default constructor
		VectorCreatorPartial() {}

		//!  Destructor
		virtual ~VectorCreatorPartial() {}

		virtual VectorInterface<T>* newVector(int d) const = 0;

		/*!
		 *  Produce a vector of size specified and populated with the data given.
		 *  This implementation uses the virtual default creator method and
		 *  subsequently initializes each element to the next value in the array given.
		 *
		 *  \param d The dimension
		 *  \param dataPtr An data array to assign each element to
		 *  \return A new vector
		 */
		virtual VectorInterface<T>* newVector(int d, const T* dataPtr) const
		{
			VectorInterface<T>* vec = this->newVector(d);
                        vec->set(d, dataPtr);
			return vec;
		}
		
		/*!
		 *  Produce a vector of size specified and constant value given.
		 *  This implementation uses the virtual default creator method and
		 *  subsequently initializes each element to the constant value given.
		 *
		 *  \param d The dimension
		 *  \param constantValue An initial value to assign each element to
		 *  \return A new vector
		 */
		virtual VectorInterface<T>* newConstantVector(int d, T constantValue) const
		{
			VectorInterface<T>* vec = this->newVector(d);
			for (int i = 0; i < d; i++)
			{
					vec->set(i, constantValue);
			}
			return vec;
		}
		
	};

		
	
	
}
}

#endif
