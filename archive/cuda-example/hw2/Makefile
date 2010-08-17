PRJ=`find . ! -name . -type d`

.PHONY: all clean emu gpu tar
all: emu
all clean emu gpu:
	for d in $(PRJ); do make $@  -C $$d -f ../cuda_build; done
tar:
	(cd ..; tar jcf cuda.tar.bz2 cuda)