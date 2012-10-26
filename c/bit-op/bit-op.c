
// 找到第一个b比x大的2的幂
int64_t next2n(const int64_t x)
{
  return x <= 2? x: (1LL << 63) >> __builtin_clz(x - 1);
}

// 判断n是否是2的n次方
int64_t is2n(const int64_t n)
{
}

// align是2的n次方
int64_t next_align(const int64_t n, const int64_t align)
{
  return (n + align - 1) & align;
}

// 返回n的二进制表示中1的个数
int count1bit(const int64_t n)
{
  return __builtin_popcountl(n);
}

// 返回第一个1的位置，注意最低位为1的话，返回结果是1，而不是0
// 找不到1的时候，返回结果为0
int first1bit(const int64_t n)
{
  return __builtin_ffsl(n);
}

// int64_t的byte反序
int64_t reverse_bytes(const int64_t n)
{
    return __builtin_bswap64(n);
}
