#ifndef __VECTOR_UBLAS_H__
#define __VECTOR_UBLAS_H__

#include "linear/lite/VectorUBlas.h"
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>


/*!
 *  \file VectorUBlas.h
 *  \brief An implementation of vector routines using uBlas
 */

namespace linear
{
	namespace lite
	{
	namespace ublas
	{
	/*!
	 *  \class VectorUBlas
	 *  \brief Implementation of VectorInterface using uBlas (part of boost)
	 *
	 *  The uBlas package can be used to interface Atlas and Lapack, but also
	 *  contains native C++ implementations for blas levels 1, 2 & 3.  This
     *  class currently uses dense uBlas vectors internally as storage for
	 *  this class.  Future versions may have a second default template argument
	 *  specifying storage (like the MTL driver, or the uBlas Matrix class).
	 */
	template<typename T> class VectorUBlas : public VectorInterface<T>
	{
	private:
		boost::numeric::ublas::vector<T> internal;
	public:
		typedef boost::numeric::ublas::vector<T> Arr_T;
		//!  Default constructor
		VectorUBlas() {}

		VectorUBlas(int n) { internal.resize(n); }

		/*!
		 *  Create a new vector initialized with a
		 *  copy of the native array type
		 *
		 *  \param m A native array to copy
		 */
		VectorUBlas(const Arr_T& m) { internal = m; }

		//!  Destructor
		virtual ~VectorUBlas() {}

		void init(int n) { internal.resize(n); }

		int size() const { return internal.size(); }

		/*!
		 *  Override the non-optimal default in the base class using uBlas-specific routines.
		 *  \param vec The vector to dot this with.
		 *  \return A dot product
		 */
		T dot(const VectorInterface<T>& vec) const
		{
			const VectorUBlas<T>& reint = down_cast< const VectorUBlas<T>& , const VectorInterface<T>& >(vec);
			return boost::numeric::ublas::inner_prod(internal, reint.internal);
		}

		/*!
		 *  Override the non-optimal default in the base class using uBlas-specific routines.
		 *
		 *  \return An L2 norm
		 */
		T norm2() const
		{
			return boost::numeric::ublas::norm_2(internal);
		}
		T get(int i) const { return internal(i); }
		//T& get(int i) { return internal(i); }
		void set(int i, const T& val) { internal[i] = val; }
		virtual VectorInterface<T>* add(const VectorInterface<T>& vec) const
		{
			const VectorUBlas<T>& reint = down_cast< const VectorUBlas<T>& , const VectorInterface<T>& >(vec);
			return new VectorUBlas<T>(internal + reint.internal);
		}


		VectorInterface<T>* cross(const VectorInterface<T>& vec) const
		{
			// replace this with an exception!
			assert(vec.size() == 3);
			assert(size() == 3);
			double u_x = get(0); double u_y = get(1); double u_z = get(2);
			double v_x = vec[0]; double v_y = vec[1]; double v_z = vec[2];
			VectorInterface<T> *xp = new VectorUBlas<T>(this->internal);
			xp->set(0, (u_y*v_z - u_z*v_y));
			xp->set(1, (u_z*v_x - u_x*v_z));
			xp->set(2, (u_x*v_y - u_y*v_x));
			return xp;
		}
		void print() const { std::cout << internal << std::endl; }
		VectorInterface<T>* clone() const
		{
			return new VectorUBlas<T>(this->internal);
		}
		/*!
		 *  Retrieve the native storage
		 */
		Arr_T& native() { return internal; }

		/*!
		 *  Retrieve the native storage (const)
		 */
		const Arr_T& native() const { return internal; }
	};

	/*!
	 *  \class VectorCreatorUBlas
	 *  \brief Creator for uBlas-backed vectors
	 *
	 *  This class currently creates only dense 1D vectors.
	 *  It uses the default implementation for constant vector generation,
	 *  which may be non-optimal.
	 *
	 *  \todo Use any available native constant array creation mechanism in uBlas
	 *  \todo Consider allowing default template parameter for storage
	 */
	template<typename T> class VectorCreatorUBlas : public VectorCreatorPartial<T>
	{
	public:
		//!  Default constructor
		VectorCreatorUBlas() {}

		//!  Destructor
		~VectorCreatorUBlas() {}

		/*!
		 *  Produce a new vector
		 *
		 */
		VectorInterface<T>* newVector(int d) const
		{
			return new math::VectorUBlas<T>(d);
		}
	};
}
}
}
#endif // __VECTOR_UBLAS_H__
