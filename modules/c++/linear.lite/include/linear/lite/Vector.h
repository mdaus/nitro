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

#ifndef __LINEAR_LITE_VECTOR_H__
#define __LINEAR_LITE_VECTOR_H__

#include "linear/lite/VectorInterface.h"
#include <iostream>
#include <memory>
/*!
 *  \file Vector.h
 *  \brief Abstractions for vector implementation
 *
 */

namespace linear
{
	namespace lite
	{
	  // Forward Declaration
	template<typename T> class Matrix;
	/*!
	 *  \class Vector
	 *  \brief Encapsulation type for vector interface
	 *
	 *  This class is the encapsulation handle for VectorInterface,
	 *  which is implemented by drivers for the various math library
	 *  implementations.
	 */

	template<typename T> class Vector
	{
	public:
	        friend class Matrix<T>;

		/*!
		 *  Constructor.  Takes a VectorInterface RHS
		 *  \param newVector The vector to contain
		 */
		Vector(VectorInterface<T>* newVector)
		{
			veci.reset(newVector);
		}

		/*!
		 *  Copy constructor.  Clones the input vector's
		 *  interface, then contains it.
		 *  \param vec The vector to copy
		 */
		Vector(const Vector& vec)
		{
			
			veci.reset(vec.veci->clone());

		}
                ~Vector() {}	
	
		/*!
		 *  Assignment operator
		 *  \param vec A vector RHS
		 *  \return This
		 */
		Vector& operator=(const Vector& vec)
		{
			// check self assign
			if (&vec != this)
			{
				veci.reset(vec.veci->clone());
			}
			return *this;
		}

		/*!
		 *  Assignment operator for newVector interface.
		 *  This is not a deep copy, its a hostile takeover
		 *  \param newVector The vector to contain
		 *  \return This
		 */
		Vector& operator=(VectorInterface<T>* newVector)
		{
			veci.reset(newVector);
			return *this;
		}
		
		/*!
		 *  Initialize vector size from an argument
		 *  \param n The argument
		 */
		void init(int n) { veci->init(n); }
		
		/*!
		 *  Retrieve the vector size
		 *  \return vector size
		 */
		int size() const { return veci->size(); }
		
		/*!
		 *  Get the dot product of this and another vector
		 *  \sa VectorInterface::dot
		 *
		 *  \param vec A vector to dot this by
		 *  \return The dot product
		 */
		T dot(const Vector& vec) const
		{
			return veci->dot(*(vec.veci));
		}
		
		/*!
		 *  Retrieve the euclidean norm
		 *  \sa VectorInterface::norm2
		 *
		 *  \return The L2 norm
		 */
		T norm2() const
		{
			return veci->norm2();
		}

                void normalize()
		{
			veci->normalize();
		}

		/*!
		 *  Get the ith element in the vector (const)
		 *  \param i The index
		 *  \return The element (a copy)
		 */
		T get(int i) const { return veci->get(i); }

		/*
		 *  Get the ith element in the vector
		 *  param i The index
		 *  return The element (a reference)
		 */
		T& get(int i) { return veci->get(i); }

                /*!
                 *  Copy the data from the vector (const) to the given pointer
                 *  The pointer must be pre-allocated
                 *  \param dataPtr The pointer to copy the vector's data to
                 */
                void get(T* dataPtr) const
                {
                   veci->get(dataPtr);
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
		 *  Rather than have a (non-const) reference operator()
		 *  overload, we provide a set method.  This is because
		 *  it appears that uBlas, and perhaps MTL do not actually
		 *  offer a sparse matrix reference to the actual raw type.
		 *  Instead, they provide an encapsulating object which can
		 *  be used in the expected manner.  Since this type cannot
		 *  be offered in the VectorInterface, we elect to hide it
		 *  entirely from the user and not allow a non-const accessor
		 *  get() or operator() method.  Hopefully, a work-around can
		 *  be devised in the near future.
		 *
		 *  \param i row index
		 *  \param elem A value to copy into the element in the specified position
		 */
		void set(int i, const T& val) { veci->set(i, val); }


                /*!
                 *  Copy the data from the given pointer to the vector
                 *  \param size The number of data elements to copy 
                 *  \param dataPtr The pointer to the data to copy
                 */
                virtual void set(int size, const T* dataPtr)
                {
                   veci->set(size, dataPtr);
                }

                /*!
                 *  Clone the vector
                 *  \return The new cloned vector
                 */
		VectorInterface<T>* clone() const
		{
		    return veci->clone();
		}

		/*!
		 *  Print the vector using the underlying driver
		 *  packages print routine
		 */
		void print() const { veci->print(); }
		
		/*!
		 *  Vector cross product
		 *  \sa VectorInterface::cross
		 *  
		 *  \param vec A vector to cross this by
		 *  \return A vector cross product
		 */
		VectorInterface<T>* cross(const Vector& vec) const
		{
			return veci->cross(*(vec.veci));
		}



		/*!
		 *  Multiply this by a scalar
		 *  \sa VectorInterface::multiply
		 *
		 *  \param scalar The amount to scale the vetor by
		 *  \return The scaled vector
		 */
		VectorInterface<T>* multiply(T scalar) const
		{
		    VectorInterface<T>* copy = veci->clone();
		    copy->scale(scalar);
		    return copy;
		}

		/*!
		 *  Add a vector to this one
		 *  \sa VectorInterface::add
		 *
		 *  \param vec The vector to add to this
		 *  \return The sum vector
		 */
		VectorInterface<T>* add(const Vector& vec) const
		{
			return veci->add(*(vec.veci));
		}
		/*!
		 *  Multiply this by a scalar, to scale
		 *  \sa VectorInterface::scale
		 *
		 *  \param scalar The amount to scale the vector by
		 */
		void scale(T scalar) 
		{
			veci->scale(scalar);
		}

		Vector&
		    operator*=(T scalar)
		{
		    scale(scalar);
		    return *this;
		}
				
		Vector& operator/=(T scalar)
		{
		    T denom(1/scalar);
		    scale(denom);
		    return *this;
		}

		

	private:
		std::auto_ptr< VectorInterface<T> > veci;
		
		Vector() {}


	};

	template<typename T> Vector<T>
	  operator*(const Vector<T>& v, T scalar)
	{
	    return v.multiply(scalar);
	}

	template<typename T> Vector<T>
	  operator*(T scalar, const Vector<T>& v)
	{
	    return v.multiply(scalar);
	}


	template<typename T> Vector<T> operator/(const Vector<T>& v, T scalar)
	{
	    T denom(1/scalar);
	    return v.multiply(denom);
	}

	template<typename T> Vector<T> operator/(T scalar, const Vector<T>& v)
	{
	    T denom(1/scalar);
	    return v.multiply(denom);
	}



	template<typename T> Vector<T> operator+(const Vector<T>& v1, 
						 const Vector<T>& v2)
	{
	    return v1.add(v2);
	}



	template<typename T> Vector<T> operator-(const Vector<T>& v1, 
						 const Vector<T>& v2)
        {
	    Vector<T> negative(v2);
	    negative.scale(-1);
	    return v1.add(negative);
	}
	
	template<typename T> std::ostream& operator<<(std::ostream& os,
						      const Vector<T>& v)
	{
	    assert(v.size());
	    os << "{";
	    int i = 0;
	    for (; i < v.size() - 1; ++i)
		os << v[i] << ","; 
		    
	    os << v[i] << "}";
	    return os;
	}


}
}
#endif
