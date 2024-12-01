#ifndef FUNC_H
#define FUNC_H

#include <cstddef>
#include <algorithm> 
#include <cctype>
#include <locale>

template <typename FROM, typename TO>
TO* cast_all(size_t _n, FROM *_a) {
    TO *_r = new TO[_n];
    for (size_t i = 0; i < _n; i++)
        _r[i] = static_cast<TO>(_a[i]);
    return _r;
}

inline bool contains(std::string &s, char c) {
    return s.find(c) != std::string::npos;
}


template <typename T>
std::vector<T> merge_vectors(const std::initializer_list<std::vector<T>>& vectors) {
    std::vector<T> result;

    // Reserve space to improve performance
    size_t total_size = 0;
    for (const auto& vec : vectors) {
        total_size += vec.size();
    }
    result.reserve(total_size);

    // Append all elements
    for (const auto& vec : vectors) {
        result.insert(result.end(), vec.begin(), vec.end());
    }

    return result;
}


template <typename T>
void print_vector(const std::vector<T>& vec, const std::string& label = "Vector") {
    std::cout << label << " (" << vec.size() << " elements): [ ";
    for (const auto& elem : vec) {
        std::cout << elem << " ";
    }
    std::cout << "]" << std::endl;
}


template <typename T>
std::vector<size_t> multi_index(std::vector<size_t> indices, T *reference) {
    std::vector<size_t> result(indices.size());
    for (size_t i = 0; i < indices.size(); i++)
        result[i] = reference[indices[i]];
    return result;
}

// Source: https://stackoverflow.com/questions/216823/how-to-trim-a-stdstring#217605
// trim from start (in place)
inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
}

// trim from end (in place)
inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

// trim from both ends (in place)
inline void trim(std::string &s) {
    rtrim(s);
    ltrim(s);
}

// trim from start (copying)
inline std::string ltrim_copy(std::string s) {
    ltrim(s);
    return s;
}

// trim from end (copying)
inline std::string rtrim_copy(std::string s) {
    rtrim(s);
    return s;
}

// trim from both ends (copying)
inline std::string trim_copy(std::string s) {
    trim(s);
    return s;
}

#endif
