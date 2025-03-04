#ifndef UNROLL_LOOP_H
#define UNROLL_LOOP_H

#include <utility>

// Source: https://stackoverflow.com/questions/37602057/why-isnt-a-for-loop-a-compile-time-expression
template<std::size_t N>
struct num { static const constexpr auto value = N; };

template <class F, std::size_t... Is>
void for_(F func, std::index_sequence<Is...>)
{
  using expander = int[];
  (void)expander{0, ((void)func(num<Is>{}), 0)...};
}

template <std::size_t N, typename F>
void for_(F func)
{
  for_(func, std::make_index_sequence<N>());
}

#endif