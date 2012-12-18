#ifndef STDINT_H
#define STDINT_H

#define __WORDSIZE 32

typedef signed char		i8;
typedef short int		i16;
typedef int			i32;
# if __WORDSIZE == 64
typedef long int		i64;
# else
__extension__
typedef long long int		i64;
# endif

typedef unsigned char		u8;
typedef unsigned short int	u16;
typedef unsigned int		u32;
#if __WORDSIZE == 64
typedef unsigned long int	u64;
#else
__extension__
typedef unsigned long long int	u64;
#endif


#endif //STDINT_H
