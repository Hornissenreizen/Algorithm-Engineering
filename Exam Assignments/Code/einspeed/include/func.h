#ifndef FUNC_H
#define FUNC_H

#include <cstddef>

template <typename FROM, typename TO>
TO* cast_all(size_t _n, FROM *_a) {
    TO *_r = new TO[_n];
    for (size_t i = 0; i < _n; i++)
        _r[i] = static_cast<TO>(_a[i]);
    return _r;
}

#endif