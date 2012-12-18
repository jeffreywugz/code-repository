#include <phi/config.h>
              
              .globl idt

              .macro isr no
isr\no:       cli
              pushl $0
              pushl $\no
              jmp   _int_handler
              .endm

              .macro isr_ no
isr\no:       cli
              pushl $\no
              jmp   _int_handler
              .endm

              .macro irq no
isr\no:       cli
              pushl $0
              pushl $\no
              jmp   _int_handler
              .endm
              
sys_call_common_stub:
              pushl $0
              pushl $0x80
              jmp _int_handler

              .section .text
              .align 4
              isr   0; isr   1; isr   2; isr   3; isr   4; isr   5; isr   6; isr   7;
              isr_  8; isr   9; isr_ 10; isr_ 11; isr_ 12; isr_ 13; isr_  14; isr  15;
              isr  16; isr  17; isr  18; isr  19; isr  20; isr  21; isr  22; isr  23;
              isr  24; isr  25; isr  26; isr  27; isr  28; isr  29; isr  30; isr  31;
              irq  32; irq  33; irq  34; irq  35; irq  36; irq  37; irq  38; irq  39;
              irq  40; irq  41; irq  42; irq  43; irq  44; irq  45; irq  46; irq  47;

              .macro save_state
              pusha
              pushl %ds
              pushl %es
              pushl %fs
              pushl %gs
              movl  $DATA_SEGMENT, %eax
              movw  %ax, %ds
              movw  %ax, %es
              movw  %ax, %fs
              movw  %ax, %gs
              movl  %esp, %eax
              subl  $4,  %eax
              pushl %eax
              .endm

              .macro restore_state
              movl  0(%esp), %eax
              movl  %eax, %esp
              popl  %eax
              popl  %gs
              popl  %fs
              popl  %es
              popl  %ds
              popa
              addl  $8, %esp
              .endm
              
_int_handler:  
              save_state
              call  int_handler
              restore_state
              iret
              
              
              .macro idt__ fun
              .long \fun
              .word CODE_SEGMENT
              .byte 0xEE
              .endm

              .macro idt_ no
              .long isr\no
              .word CODE_SEGMENT
              .byte 0x8E
              .endm
              
              .data
              .align 32
idt:
              idt_  0; idt_  1; idt_  2; idt_  3; idt_  4; idt_  5; idt_  6; idt_  7;
              idt_  8; idt_  9; idt_ 10; idt_ 11; idt_ 12; idt_ 13; idt_ 14; idt_ 15;
              idt_ 16; idt_ 17; idt_ 18; idt_ 19; idt_ 20; idt_ 21; idt_ 22; idt_ 23;
              idt_ 24; idt_ 25; idt_ 26; idt_ 27; idt_ 28; idt_ 29; idt_ 30; idt_ 31;
              idt_ 32; idt_ 33; idt_ 34; idt_ 35; idt_ 36; idt_ 37; idt_ 38; idt_ 39;
              idt_ 40; idt_ 41; idt_ 42; idt_ 43; idt_ 44; idt_ 45; idt_ 46; idt_ 47;
              .skip 7*(128-48);
              idt__ sys_call_common_stub;     .skip 7*127;
