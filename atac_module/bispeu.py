import numpy as np
import ctypes
from ctypes import pythonapi
import numba
from numba import types
from numba.extending import intrinsic
from numba.core import cgutils
import scipy.interpolate.dfitpack

def capsule_name(capsule):
    """ from https://github.com/numba/numba/issues/7818 """
    pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
    pythonapi.PyCapsule_GetName.argtypes = [ctypes.py_object]
    return pythonapi.PyCapsule_GetName(capsule)

def get_f2py_function_address(capsule):
    """ from https://github.com/numba/numba/issues/7818 """
    name = capsule_name(capsule)
    pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    return pythonapi.PyCapsule_GetPointer(capsule, name)

@intrinsic
def val_to_ptr(typingctx, data):
    def impl(context, builder, signature, args):
        ptr = cgutils.alloca_once_value(builder,args[0])
        return ptr
    sig = types.CPointer(numba.typeof(data).instance_type)(numba.typeof(data).instance_type)
    return sig, impl

@intrinsic
def ptr_to_val(typingctx, data):
    def impl(context, builder, signature, args):
        val = builder.load(args[0])
        return val
    sig = data.dtype(types.CPointer(data.dtype))
    return sig, impl

_bispeu_functype = ctypes.CFUNCTYPE(ctypes.c_void_p,
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_longlong),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_longlong),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_longlong))

BISPEU = _bispeu_functype(get_f2py_function_address(scipy.interpolate.dfitpack.bispeu._cpointer))

@numba.njit('float64[::1](float64[::1],float64[::1],float64[::1],int64,int64,float32[::1],float32[::1])')
def bispeu_wrapped(tx, ty, c, kx, ky, x, y):
    z = np.empty(x.shape[0], dtype=np.float64)
    nx_arr = val_to_ptr(numba.int64(tx.shape[0]))
    ny_arr = val_to_ptr(numba.int64(ty.shape[0]))
    kx_arr = val_to_ptr(numba.int64(kx))
    ky_arr = val_to_ptr(numba.int64(ky))
    m_arr = val_to_ptr(numba.int64(x.shape[0]))
    lwrk_arr = val_to_ptr(numba.int64(kx+ky+2))
    wrk = np.zeros(kx+ky+2, dtype=np.float64)
    ier_arr = val_to_ptr(numba.int64(0))
    BISPEU(tx.ctypes, nx_arr, ty.ctypes, ny_arr, c.ctypes, kx_arr, ky_arr,
           x.astype(np.float64).ctypes, y.astype(np.float64).ctypes, z.ctypes, m_arr, wrk.ctypes, lwrk_arr, ier_arr)
    return z
