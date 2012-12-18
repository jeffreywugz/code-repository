#include <phi/config.h>
              
              .globl startup
              .text
              .align 4
#include "mboot.inc"
startup:
              movl  $stack, %esp
              pushl %eax
              addl  $LOAD_OFFSET, %ebx
              pushl %ebx
              call  setup
              addl  $LOAD_OFFSET, %esp
              call  kmain
              
              .section .bss
              .align 32
              .skip KSTACK_SIZE
stack:
