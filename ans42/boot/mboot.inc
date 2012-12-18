MODULEALIGN =  1<<0
MEMINFO     =  1<<1
FLAGS       =  MODULEALIGN | MEMINFO
MAGIC       =    0x1BADB002
CHECKSUM    = -(MAGIC + FLAGS)

MultiBootHeader:
       .long MAGIC
       .long FLAGS
       .long CHECKSUM
