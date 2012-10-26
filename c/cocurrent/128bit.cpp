#include <stdint.h>

namespace types
{
  struct uint128_t
  {
    uint64_t lo;
    uint64_t hi;
  }
  __attribute__ (( __aligned__( 16 ) ));
}

template< class T > inline bool cas( volatile T * src, T cmp, T with );

template<> inline bool cas( volatile types::uint128_t * src, types::uint128_t cmp, types::uint128_t with )
{
  bool result;
  __asm__ __volatile__
    (
      "\n\tlock cmpxchg16b %1"
      "\n\tsetz %0\n"
      : "=q" ( result )
        , "+m" ( *src )
        , "+d" ( cmp.hi )
        , "+a" ( cmp.lo )
      : "c" ( with.hi )
        , "b" ( with.lo )
      : "cc"
      );
  return result;
}

int main()
{
  using namespace types;
  uint128_t test = { 0xdecafbad, 0xfeedbeef };
  uint128_t cmp = test;
  uint128_t with = { 0x55555555, 0xaaaaaaaa };
  return ! cas( & test, cmp, with );
}
