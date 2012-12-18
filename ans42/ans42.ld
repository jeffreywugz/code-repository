OUTPUT_FORMAT("elf32-i386")
OUTPUT_ARCH(i386)
ENTRY(startup)

LOAD_PHY_ADDR = 0x100000;
LOAD_USR_PHY_ADDR = 0x8000000;
LOAD_OFFSET = 0xc0000000;
SEGMENT_SIZE = 0x400000;
PAGE_SIZE = 0x1000;
CACHELINE_SIZE = 0x20;

PHDRS
{
    boot PT_LOAD FLAGS(5);  /* R_E */
    text PT_LOAD FLAGS(5);  /* R_E */
    data PT_LOAD FLAGS(7);  /* RWE */
    user PT_LOAD FLAGS(7);  /* RWE */
}

SECTIONS
{
    . = 0;
    .user : AT(ADDR(.user)+LOAD_USR_PHY_ADDR) {
          _usr_data_start = .;
         usr\/*(.text*)
         usr\/*(.rodata*)
         usr\/*(COMMON*)
         usr\/*(.data*)
         usr\/*(.bss*)
         _usr_data_end = ALIGN(SEGMENT_SIZE);
    } :user

    . = LOAD_PHY_ADDR;
    .boot : AT(ADDR(.boot)) {
         _start_phy = .;
         boot\/*(.text*)
         boot\/*(.rodata*)
         boot\/*(COMMON*)
         boot\/*(.data*)
         boot\/*(.bss*)
    } :boot = 0x9090
    
    . +=  LOAD_OFFSET;
    .text.init ALIGN(PAGE_SIZE): AT(ADDR(.text.init) - LOAD_OFFSET) {
         init\/*(.text*)
         init\/*(.rodata*)
         init\/*(COMMON*)
         init\/*(.data*)
         init\/*(.bss*)
    } :text = 0x9090

    .text ALIGN(PAGE_SIZE): AT(ADDR(.text) - LOAD_OFFSET) {
         text = .;
         *(.text*)
         *(.rodata*)
         _etext = .;
    } :text = 0x9090
  
    .data ALIGN(PAGE_SIZE): AT(ADDR(.data) - LOAD_OFFSET) {
        *(.data.gdt)
        *(.data)
        *(COMMON)
    } :data
  
    .data.page_aligned ALIGN(PAGE_SIZE): AT(ADDR(.data.page_aligned) - LOAD_OFFSET) {
        *(.data.page_aligned)
        *(.data.page_dir)
        *(.data.idt)
    } :data
  
    .data.cacheline_aligned ALIGN(CACHELINE_SIZE): AT(ADDR(.data.cacheline_aligned) - LOAD_OFFSET) {
        *(.data.cacheline_aligned)
        _edata = .;	
    } :data
  
    .bss ALIGN(PAGE_SIZE): AT(ADDR(.bss) - LOAD_OFFSET) {
        *(.bss.page_aligned)
        *(.bss)
    } :data
    
     _end = ALIGN(PAGE_SIZE);
    
    /DISCARD/ : {
        *(.comment*)
        *(.note*)
    }
}