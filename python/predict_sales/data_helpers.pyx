import numpy as np
cimport numpy as np
cimport cython

np.import_array()


DTYPE = np.int64
ctypedef np.int64_t DTYPE_t

def is_promo2_active(DTYPE_t promo2, DTYPE_t start_year,
                     DTYPE_t start_week, np.ndarray[DTYPE_t] date_year,
                     np.ndarray[DTYPE_t] date_week,
                     np.ndarray[DTYPE_t] date_month, object interval):
    cdef int N = len(date_year)
    cdef np.ndarray[DTYPE_t] promo2_active = np.zeros(N, dtype=DTYPE)
    cdef int i
    for i in range(N):
        if promo2:
            if ((date_year[i] > start_year)
                or ((date_year[i] == start_year) and (date_week[i] >= start_week))):
                if date_month[i] in interval:
                    promo2_active[i] = 1

    return promo2_active
