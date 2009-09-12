#ifndef __VECTOR_MTL_H__
#define __VECTOR_MTL_H__

#include "linear/lite/VectorInterface.h"
#include "mtl/mtl.h"

/*!
 *  \file VectorMTL.h
 *  \brief MTL driver implementation of a VectorInterface
 *
 *
 */

namespace linear
{
namespace lite
{
namespace mtl
{
	/*!
	 *  \class VectorMTL
	 *  \brief vector Matrix Template Library (MTL) driver
	 *
	 *  This class is an implementation of the VectorInterface using the Matrix Template Library (MTL).
	 *  It adds an optional template parameter for internal storage type which defaults to a densely
	 *  stored array.
	 *
	 */
	template<typename T, typename Storage_T = mtl::dense1D<T> > class VectorMTL : public VectorInterface<T>
	{

	private:
		Storage_T internal;
	public:

		//!  Default constructor
		VectorMTL() {}

		/*!
		 *  Construct an MTL vector from internal storage.
		 *  Deep copies argument
		 *
		 *  \param storage An argument to copy this vector from
		 *
		 */
		VectorMTL(const Storage_T& storage)
		{
			internal = storage;
			internal = Storage_T(storage.size());
			mtl::copy(storage, internal);
		}

		VectorMTL(int n) : VectorInterface<T>(n) { internal = Storage_T(n); }


		~VectorMTL() {}
		void init(int n) { internal = Storage_T(n); }

		int size() const { return internal.size(); }

		/*!
		 *  Add two vectors (this and the argument).  This requires casting the argument to a VectorMTL
		 *
		 *  \param vec A VectorMTL
		 *  \return The sum of the two vectors
		 */
		VectorInterface<T>* add(const VectorInterface<T>& vec) const
		{
			const VectorMTL<T>& reint = down_cast<const VectorMTL<T>&, const VectorInterface<T>& >(vec);
			VectorMTL<T, Storage_T>* sum = new VectorMTL<T, Storage_T>(internal);
			mtl::add(reint.internal, sum->internal);
			return sum;
		}
		/*!
		 *  Scale.  Overrides default implementation in favor of MTL-specific version.
		 *  \param scalar What to scale by
		 */
		void scale(T scalar)
		{
			mtl::scale(internal, scalar);
		}
		/*!
		 *  3-D vector cross product.  Currently asserts the number of dimensions
		 *
		 *  \param vec What to cross this with
		 *  \return Vector cross product
		 */
		VectorInterface<T>* cross(const VectorInterface<T>& vec) const
		{
			// replace this with an exception!
			assert(vec.size() == 3);
			assert(size() == 3);

			double u_x = get(0); double u_y = get(1); double u_z = get(2);
			double v_x = vec[0]; double v_y = vec[1]; double v_z = vec[2];
			VectorInterface<T> *xp = new VectorMTL<T, Storage_T>(this->internal);
			xp->set(0, (u_y*v_z - u_z*v_y));
			xp->set(1, (u_z*v_x - u_x*v_z));
			xp->set(2, (u_x*v_y - u_y*v_x));

			return xp;
		}
		/*!
		 *  Dot product.  Overrides default implementation in favor of MTL-specific version.
		 *
		 *  \param vec A VectorMTL
		 *  \return A dot product
		 */
		T dot(const VectorInterface<T>& vec) const
		{
			//assert(typeid(vec) ==
			const VectorMTL<T, Storage_T>& reint = down_cast< const VectorMTL<T, Storage_T>&, const VectorInterface<T>& >(vec);
			return mtl::dot(internal, reint.internal);
		}

		/*!
		 *  L2 Norm.  Overrides default implementation in favor of MTL-specific version.
		 *
		 *  \return A norm
		 */
		T norm2() const
		{
			return mtl::two_norm(internal);
		}
		T get(int i) const { return internal[i]; }
		//T& get(int i) { return internal[i]; }
		void set(int i, const T& val) { internal[i] = val; }
		void print() const { mtl::print_vector(internal); }

		/*!
		 *  Clone a vector.  New vector shares same basic type of storage
		 *  \return A new vector deep copy
		 */
		VectorInterface<T>* clone() const
		{
			return new VectorMTL<T, Storage_T>(this->internal);
		}

		/*!
		 *  Get native storage (non-const)
		 */
		Storage_T& native() { return internal; }
		/*!
		 *  Get native storage (const)
		 */
		const Storage_T& native() const { return internal; }
	};

	/*!
	 *  \class VectorCreatorMTL
	 *  \brief Implements creator pattern for MTL vectors
	 *
	 *  This class fully implements the creator pattern, starting with the existing partial implementation,
	 *  and customizing the default creation method.  Non-default storage is requested through specializing
	 *  the second template parameter to a non-default MTL 1D vector.
	 */
	template<typename T, typename Storage_T = mtl::dense1D<T> > class VectorCreatorMTL : public VectorCreatorPartial<T>
	{
	public:
		//!  Deafult constructor
		VectorCreatorMTL() {}

		//!  Destructor
		~VectorCreatorMTL() {}

		/*!
		 *  Spawn a new vector (MTL), of size given
		 *  \param d The dimension of the vector
		 *  \return The new vector
		 */
		VectorInterface<T>* newVector(int d) const
		{
			return new VectorMTL<T, Storage_T>(d);
		}
	};
}
}
}
#endif //__VECTOR_MTL_H__
