%module math_poly

%feature("autodoc", "1");

%{
#include <string>
#include <sstream>
#include "math/poly/OneD.h"
#include "math/poly/TwoD.h"
%}

%import "coda_except.i"

%include "std_string.i"

%exception
{
    try
    {
        $action
    } 
    catch (const std::exception& e)
    {
        if (!PyErr_Occurred())
        {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    }
    catch (const except::Exception& e)
    {
        if (!PyErr_Occurred())
        {
            PyErr_SetString(PyExc_RuntimeError, e.getMessage().c_str());
        }
    }
    catch (...)
    {
        if (!PyErr_Occurred())
        {
            PyErr_SetString(PyExc_RuntimeError, "Unknown error");
        }
    }
    if (PyErr_Occurred())
    {
        SWIG_fail;
    }
}

%include "math/poly/OneD.h"

%template(Poly1D) math::poly::OneD<double>;


%extend math::poly::OneD<double>
{
    double __getitem__(long i)
    { 
        if (i > self->order())
        {
            PyErr_SetString(PyExc_ValueError, "Index out of range");
            return 0.0;
        }
        return (*self)[i];
    }
    
    void __setitem__(long i, double val)
    { 
        if (i > self->order())
        {
            PyErr_SetString(PyExc_ValueError, "Index out of range");
            return;
        }
        (*self)[i] = val;
    }
    
    std::string __str__()
    {
        std::ostringstream ostr;
        ostr << *self;
        return ostr.str();
    }
}

%include "math/poly/TwoD.h"

%template(Poly2D) math::poly::TwoD<double>;

%extend math::poly::TwoD<double>
{
    double __getitem__(PyObject* inObj)
    {
        if (!PyTuple_Check(inObj))
        {
            PyErr_SetString(PyExc_TypeError, "Expecting a tuple");
            return 0.0;
        }
        Py_ssize_t xpow, ypow;
        if (!PyArg_ParseTuple(inObj, "nn", &xpow, &ypow))
        {
            return 0.0;
        }
        if (xpow > self->orderX() || ypow > self->orderY())
        {
            PyErr_SetString(PyExc_ValueError, "Index out of range");
            return 0.0;
        }
        return (*self)[xpow][ypow];
    }

    void __setitem__(PyObject* inObj, double val)
    {
        if (!PyTuple_Check(inObj))
        {
            PyErr_SetString(PyExc_TypeError, "Expecting a tuple");
            return;
        }
        Py_ssize_t xpow, ypow;
        if (!PyArg_ParseTuple(inObj, "nn", &xpow, &ypow))
        {
            return;
        }
        if (xpow > self->orderX() || ypow > self->orderY())
        {
            PyErr_SetString(PyExc_ValueError, "Index out of range");
            return;
        }
        (*self)[xpow][ypow] = val;
    }

    std::string __str__()
    {
        std::ostringstream ostr;
        ostr << *self;
        return ostr.str();
    }
}
