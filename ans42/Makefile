prj.name = ans42
prj.version = 0.1
prj.modules = boot arch init task mm sys dev lib usr
prj.bak-dir = ~/arc

include-path = include
CPP = cpp -I$(include-path)
SPP = cpp -x assembler-with-cpp -I$(include-path)
CC = gcc
CFLAGS = -Wall -Werror -g -m32 -march=i386 -oformart=i386 -fstrength-reduce -fomit-frame-pointer -finline-functions -nostdinc -ffreestanding -I$(include-path)
AS = gcc -x assembler-with-cpp
ASFLAGS = -g -m32 -march=i386 -oformat=elf-i386 -I$(include-path)
LD = ld
LDFLAGS = -r -melf_i386 --oformat elf32-i386

.PHONY: all run clean realclean checkout commit bak dep etags cscope

all: run

run: floppy
	qemu -vga std -k en-us -m 256 -fda $< -boot a
debug: floppy
	qemu -s -S -vga std -k en-us -m 256 -fda $< -boot a

prj.kobjs = $(foreach name,$(prj.modules),$(name)/$(name).ko)
$(prj.name).img: $(prj.kobjs) $(prj.name).ld
	$(LD) -Map $(prj.name).map -T $(prj.name).ld -o $@ $(prj.kobjs)
	objdump -S $(prj.name).img >$(prj.name).s
floppy: $(prj.name).img
	if [ ! -e $@ ];then bunzip2 -k $@.bz2; fi
	sudo mount $@ mnt -o loop
	sudo cp $< mnt
	sudo umount mnt
	touch $@

define module-setup
$(1)/$(1).ko: $(patsubst %.c,%.o,$(wildcard $(1)/*.c)) $(patsubst %.s,%.o,$(wildcard $(1)/*.s))
	$(LD) $(LDFLAGS) -o $$@ $$^
endef
$(foreach m,$(prj.modules),$(eval $(call module-setup,$(m))))
%.o: %.c
	$(CC) -c  $(CFLAGS) -o $@ $<
%.o: %.s
	$(AS) -c $(ASFLAGS) -o $@ $<

clean:
	rm -rf $(prj.name).img $(prj.name).map $(prj.name).s $(prj.kobjs) $(foreach m,$(prj.modules),$(m)/*.o)
realclean: clean
	rm -rf  hd TAGS cscope.files cscope.out .dep
etags:
	etags.emacs `find . -type f -name "*.[ch]"`
cscope:
	find . -type f -name "*.[ch]" >cscope.files
	cscope -b -k -I include -f cscope.out -i cscope.files
checkout:
	(cd ..; svn checkout https://prj.googlecode.com/svn/trunk/ ans42 --username huafengxi)
commit: realclean
	svn commit  .
bak: clean
	(cd ..; tar jcf $(prj.name).tar.bz2 $(prj.name); mv $(prj.name).tar.bz2 $(prj.bak-dir))
dep:
	(for f in $(foreach m,$(prj.modules),$(wildcard $(m)/*.c)); do $(CPP) -M -MM -MG -MT $${f/.c/.o} $$f; done; \
	 for f in $(foreach m,$(prj.modules),$(wildcard $(m)/*.s)); do $(CPP) -M -MM -MG -MT $${f/.s/.o} $$f; done;) >.dep
-include .dep
