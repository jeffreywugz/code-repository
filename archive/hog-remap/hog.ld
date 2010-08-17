PAGE_SIZE = 0x1000;

SECTIONS
{
    _hog_start_ = ALIGN(PAGE_SIZE);
    .hog ALIGN(PAGE_SIZE): {
         *(.hog)
    }
    _hog_end_ = ALIGN(PAGE_SIZE);
}
INSERT AFTER .bss;
