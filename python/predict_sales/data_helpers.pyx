import numpy as np
cimport numpy as np
cimport cython

np.import_array()


DTYPE = np.int64
ctypedef np.int64_t DTYPE_t

def is_promo2_active_cython(np.ndarray[DTYPE_t] promo2, np.ndarray[DTYPE_t] start_year, 
                            np.ndarray[DTYPE_t] start_week, np.ndarray[DTYPE_t] date_year,
                            np.ndarray[DTYPE_t] date_week,
                            np.ndarray[DTYPE_t] date_month, np.ndarray[object] interval):
    cdef int N = len(promo2)
    cdef np.ndarray[DTYPE_t] promo2_active = np.zeros(N, dtype=DTYPE)
    cdef int i
    for i in range(N):
        if promo2[i]:
            if ((date_year[i] > start_year[i])
                or ((date_year[i] == start_year[i]) and (date_week[i] >= start_week[i]))):
                if date_month[i] in interval[i]:
                    promo2_active[i] = 1

    return promo2_active