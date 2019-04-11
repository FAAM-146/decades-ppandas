#!python
#cython: language_level=3

from libc.math cimport sin, cos, tan, sqrt, pi

import numpy as np
import array
from cpython cimport array
cimport numpy as np
cimport cython

cdef double deg2rad(double deg):
    return deg * pi / 180.

cdef double sind(double deg):
    return sin(deg2rad(deg))

cdef double cosd(double deg):
    return cos(deg2rad(deg))

cdef double tand(double deg):
    return tan(deg2rad(deg))

@cython.boundscheck(False)
cdef void c_winds_matv(double[3][3] a, double[3] b, double[3] c) nogil:
    cdef double t[3]
    t[0] = a[0][0] * b[0] + a[0][1] * b[1] + a[0][2] * b[2]
    t[1] = a[1][0] * b[0] + a[1][1] * b[1] + a[1][2] * b[2]
    t[2] = a[2][0] * b[0] + a[2][1] * b[1] + a[2][2] * b[2]
    c[0] = t[0]
    c[1] = t[1]
    c[2] = t[2]

@cython.boundscheck(False)
cdef void c_winds_mulm(double[3][3] a, double[3][3] b, double[3][3] c) nogil:
    cdef double t2[3][3]
    t2[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0]
    t2[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1]
    t2[0][2] = a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2]
    t2[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0]
    t2[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1]
    t2[1][2] = a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2]
    t2[2][0] = a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0]
    t2[2][1] = a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1]
    t2[2][2] = a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2]
    c[0][0] = t2[0][0]
    c[1][0] = t2[1][0]
    c[2][0] = t2[2][0]
    c[0][1] = t2[0][1]
    c[1][1] = t2[1][1]
    c[2][1] = t2[2][1]
    c[0][2] = t2[0][2]
    c[1][2] = t2[1][2]
    c[2][2] = t2[2][2]

@cython.boundscheck(False)
cdef void c_winds_vadd(double[3] a, double[3] b, double[3] c) nogil:
    cdef double t[3]
    t[0] = a[0] + b[0]
    t[1] = a[1] + b[1]
    t[2] = a[2] + b[2]
    c[0] = t[0]
    c[1] = t[1]
    c[2] = t[2]

@cython.boundscheck(False)
cdef void c_winds_vmul(double[3] a, double[3] b, double[3] c) nogil:
    cdef double t[3]
    t[0] = a[1] * b[2] - a[2] * b[1]
    t[1] = a[2] * b[0] - a[0] * b[2]
    t[2] = a[0] * b[1] - a[1] * b[0]
    c[0] = t[0]
    c[1] = t[1]
    c[2] = t[2]

@cython.cdivision(True)
@cython.boundscheck(False)
cdef tuple _winds(double[:] tas, double[:] aoa, double[:] aos, double[:] vn, double[:] ve, double[:] vz,
                  double[:] hdg, double[:] pit, double[:] rol, double l, double m, double n,
                  double[:] yawr, double[:] pitr, double[:] rolr):

    cdef Py_ssize_t i

    cdef double [:] u = np.zeros_like(tas)
    cdef double [:] v = np.zeros_like(tas)
    cdef double [:] w = np.zeros_like(tas)

    cdef double rp[3]

    cdef double ra1[3][3]
    cdef double ra2[3][3]
    cdef double ra3[3][3]

    cdef double rua[3]

    cdef double rvg[3]
    cdef double ryr[3]
    cdef double rpr[3]
    cdef double rrr[3]
    cdef double rwind[3]
    cdef double rtemp[3]

    cdef double rtmp[3][3]
    cdef double rt[3][3]

    cdef double tan_aoa
    cdef double tan_aos
    cdef double d

    for i in range(tas.shape[0]):
        rtemp[0] = 0.
        rtemp[1] = 0.
        rtemp[2] = 0.

        rp[0] = l
        rp[1] = m
        rp[2] = n

        ra1[0][0] = 1.
        ra1[0][1] = 0.
        ra1[0][2] = 0.
        ra1[1][0] = 0.
        ra1[1][1] = cosd(rol[i])
        ra1[1][2] = -sind(rol[i])
        ra1[2][0] = 0.
        ra1[2][1] = sind(rol[i])
        ra1[2][2] = cosd(rol[i])

        ra2[0][0] = cosd(pit[i])
        ra2[0][1] = 0.
        ra2[0][2] = -sind(pit[i])
        ra2[1][0] = 0.
        ra2[1][1] = 1.
        ra2[1][2] = 0.
        ra2[2][0] = sind(pit[i])
        ra2[2][1] = 0.
        ra2[2][2] = cosd(pit[i])

        ra3[0][0] = cosd(hdg[i])
        ra3[0][1] = sind(hdg[i])
        ra3[0][2] = 0.
        ra3[1][0] = -sind(hdg[i])
        ra3[1][1] = cosd(hdg[i])
        ra3[1][2] = 0.
        ra3[2][0] = 0.
        ra3[2][1] = 0.
        ra3[2][2] = 1.

        tan_aoa = tand(aoa[i])
        tan_aos = tand(aos[i])
        d = sqrt(1. + tan_aos**2 + tan_aoa**2)
        rua[0] = -tas[i] / d
        rua[1] = -tas[i] * tan_aos / d
        rua[2] = tas[i] * tan_aoa / d

        rvg[0] = vn[i]
        rvg[1] = -ve[i]
        rvg[2] = -vz[i]

        ryr[0] = 0.
        ryr[1] = 0.
        ryr[2] = -yawr[i] *  pi / 180.

        rpr[0] = 0.
        rpr[1] = -pitr[i] * pi / 180.
        rpr[2] = 0.

        rrr[0] = rolr[i] * pi / 180.
        rrr[1] = 0.
        rrr[2] = 0.

        rwind[0] = 0.
        rwind[1] = 0.
        rwind[2] = 0.

        c_winds_mulm(ra3, ra2, rtmp)
        c_winds_mulm(rtmp, ra1, rt)
        c_winds_matv(rt, rua, rua)
        c_winds_vadd(rua, rwind, rwind)
        c_winds_vadd(rvg, rwind, rwind)
        c_winds_matv(rt, rp, rp)
        c_winds_mulm(ra3, ra2, rtmp)
        c_winds_matv(rtmp, rrr, rrr)
        c_winds_matv(ra3, rpr, rtemp)
        c_winds_vadd(rrr, rtemp, rtemp)
        c_winds_vadd(ryr, rtemp, rtemp)
        c_winds_vmul(rtemp, rp, rtemp)
        c_winds_vadd(rtemp, rwind, rwind)

        u[i] = rwind[0]
        v[i] = -rwind[1]
        w[i] = rwind[2]

    return v, u, w


cpdef c_winds(double[:] tas, double[:] aoa, double[:] aos, double[:] vn, double[:] ve, double[:] vz,
              double[:] hdg, double[:] pit, double[:] rol, double l, double m, double n,
              double[:] yawr, double[:] pitr, double[:] rolr):
    return _winds(tas, aoa, aos, vn, ve, vz, hdg, pit, rol, l, m, n, yawr, pitr, rolr)
