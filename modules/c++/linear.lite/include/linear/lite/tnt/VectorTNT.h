#ifndef __VECTOR_TNT_H__
#define __VECTOR_TNT_H__

#include "linear/lite/VectorInterface.h"
#include "tnt_array1d.h"

/*!
 *  \file VectorTNT.h
 *  \brief A vector implementation using TNT
 *
 *
 */

namespace linear
{
	namespace lite
	{
	namespace tnt
	{

	/*!
	 *  \class VectorTNT
	 *  \brief A vector implementation using Templated Numerical Toolkit (TNT)
	 *
	 *  This class implements the VectorInterface using TNT.  Since TNT recently deprecated
	 *  its vector API, this class is implemented using an underlying TNT::Array.  For this
	 *  reason, the TNT::Vector internal methods were not used in this class, so it relies
	 *  heavily on the VectorInterface class' non-optimal, default implementations.
	 *
	 */
	template<typename T> class VectorTNT : public VectorInterface<T>
	{
	private:
		TNT::Array1D<T> internal;
	public:

		//typedef TNT::Array1D<typename T> Arr_T;
                typedef TNT::Array1D<T> Arr_T;

		// Default constructor.
		VectorTNT() {}

		VectorTNT(int n) { internal = Arr_T(n); }

		/*!
		 *  Deep copy from a native array.  Uses internal copy() method
		 *  in native type.
		 *  \param a A native array to copy
		 */
		VectorTNT(const Arr_T& a)
		{
			internal = a.copy();
		}
		//VectorTNT(const TNT::Array2D& a) { internal = a.copy(); }
		virtual ~VectorTNT() {}
		void init(int n) { internal = Arr_T(n); }

		int size() const { return internal.dim1(); }

		T get(int i) const { return internal[i]; }
		void set(int i, const T& val) { internal[i] = val; }
		T& get(int i) { return internal[i]; }
		VectorInterface<T>* add(const VectorInterface<T>& vec) const
		{
			assert(size() == vec.size());
			VectorInterface<T>* sum = new VectorTNT<T>(vec.size());
			for (int i = 0; i < size(); i++)
			{
				sum->set(i, (get(i) + vec.get(i)));
			}
			return sum;
		}

		/*!
		 *  3D cross product (only).
		 *
		 *  \param vec The argument to cross this by
		 *  \return Vector cross product (TNT)
		 */

		VectorInterface<T>* cross(const VectorInterface<T>& vec) const
		{
			// replace this with an exception!
			assert(vec.size() == 3);
			assert(size() == 3);
			double u_x = get(0); double u_y = get(1); double u_z = get(2);
			double v_x = vec[0]; double v_y = vec[1]; double v_z = vec[2];
			VectorInterface<T> *xp = new VectorTNT<T>(this->internal);
			xp->set(0, (u_y*v_z - u_z*v_y));
			xp->set(1, (u_z*v_x - u_x*v_z));
			xp->set(2, (u_x*v_y - u_y*v_x));
			return xp;
		}

		/*!
		 *  Clone using the native copy constructor
		 *  \return A deep copy
		 */
		VectorInterface<T>* clone() const
		{
			return new VectorTNT<T>(this->internal);
		}
		void print() const
		{
		  std::cout << "{";
		  int i = 0;
		  for (; i < size() - 1; ++i)
		    std::cout << internal[i] << ",";

		  if (size())
		    std::cout << internal[i];
		  std::cout << "}" << std::endl;
		}

		/*!
		 *  Retrieve a native TNT type
		 */
		Arr_T& native() { return internal; }
		/*!
		 *  Retrieve a native TNT type (const)
		 */
		const Arr_T& native() const { return internal; }
	};

	/*!
	 *  \class VectorCreatorTNT
	 *  \brief Creates TNT-backed vectors using VectorCreatePartial as a base.
	 *
	 *  This class overrides VectorCreatorPartial given the implementation for
	 *  the default creator method.
	 */
	template<typename T> class VectorCreatorTNT : public VectorCreatorPartial<T>
	{
	public:
		//!  Constructor
		VectorCreatorTNT() {}
		//!  Destructor
		~VectorCreatorTNT() {}

		/*!
		 *  Create a new vector which is TNT backed, with dimension given
		 *  \param d The dimension of the vector
		 *  \return A vector of type and dimension specified
		 */
		VectorInterface<T>* newVector(int d) const
		{
		    return new VectorTNT<T>(d);
		}

                VectorInterface<T>* newVector(int d, const T* dataPtr) const
                {
                    return VectorCreatorPartial<T>::newVector(d, dataPtr);
                }

	};
}
}
}
#endif //__VECTOR_TNT_H__
