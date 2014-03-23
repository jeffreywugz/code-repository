/* opcodes taken from objdump of "crc32b (%%rdx), %%rcx"
   for RHEL4 support (GCC 3 doesn't support this instruction) */
#define crc32_sse42_byte                                \
  asm(".byte 0xf2, 0x48, 0x0f, 0x38, 0xf0, 0x0a"        \
      : "=c"(crc) : "c"(crc), "d"(buf));                \
  len--, buf++

/* opcodes taken from objdump of "crc32q (%%rdx), %%rcx"
   for RHEL4 support (GCC 3 doesn't support this instruction) */
#define crc32_sse42_quadword                            \
  asm(".byte 0xf2, 0x48, 0x0f, 0x38, 0xf1, 0x0a"        \
      : "=c"(crc) : "c"(crc), "d"(buf));                \
  len -= 8, buf += 8

inline static uint64_t crc64_sse42(uint64_t uCRC64, 
                                   const char* buf, int64_t len)
{
  uint64_t crc = uCRC64;

  if (NULL != buf && len > 0)
  {
    while (len && ((uint64_t) buf & 7)) {
      crc32_sse42_byte;
    }

    while (len >= 32) {
      crc32_sse42_quadword;
      crc32_sse42_quadword;
      crc32_sse42_quadword;
      crc32_sse42_quadword;
    }

    while (len >= 8) {
      crc32_sse42_quadword;
    }

    while (len) {
      crc32_sse42_byte;
    }
  }

  return crc;
}

