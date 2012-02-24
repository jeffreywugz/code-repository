prj=obcon
cxx = g++
top-dir = ../..
std-header-defs = -DSTDC_HEADERS=1 -DHAVE_SYS_TYPES_H=1 -DHAVE_SYS_STAT_H=1 -DHAVE_STDLIB_H=1 -DHAVE_STRING_H=1 -DHAVE_MEMORY_H=1 -DHAVE_STRINGS_H=1 -DHAVE_INTTYPES_H=1 -DHAVE_STDINT_H=1 -DHAVE_UNISTD_H=1 -DHAVE_DLFCN_H=1 -D__STDC_LIMIT_MACROS
package-defs = -DPACKAGE_NAME=\"OceanBase\" -DPACKAGE_TARNAME=\"oceanbase\" -DPACKAGE_VERSION=\"0.2.5\" -DPACKAGE_STRING=\"OceanBase\ 0.2.5\" -DPACKAGE_BUGREPORT=\"qushan@taobao.com\"  -DPACKAGE=\"oceanbase\" -DVERSION=\"0.2.5\" -DRELEASEID=\"\"
inc-flags = -I. -I$$TBLIB_ROOT/include/tbsys -I$$TBLIB_ROOT/include/tbnet -I$(top-dir)/include -I$(top-dir)/src
normal-cxx-flags = -g -Wall -DORDER_SERIALIZE -DCOMPATIBLE
cxx-flags= $(package-defs) $(std-header-defs) $(inc-flags) $(normal-cxx-flags)
ldflags = -Wall -g -lpthread -lc -lm -lrt -ldl -laio -lnuma $(top-dir)/src/updateserver/libupdateserver.a $(top-dir)/src/rootserver/librootserver.a $(top-dir)/src/chunkserver/libchunkserver.a $(top-dir)/src/mergeserver/libmergeserver.a $(top-dir)/src/sstable/libsstable.a $(top-dir)/src/common/libcommon.a $(top-dir)/src/common/compress/libcomp.a $(top-dir)/src/common/btree/libbtree.a $$TBLIB_ROOT/lib/libtbnet.a $$TBLIB_ROOT/lib/libtbsys.a

all: $(prj)
	./$(prj)

%.o: %.cpp
	$(cxx) -c $< -o $@ $(cxx-flags)
srcs= $(wildcard *.cpp)
objs= $(patsubst %.cpp,%.o,$(srcs))
$(prj): $(objs)
	$(cxx) -o $@ $^ $(ldflags)
%.o: %.c
	$(cxx) -c  $(cxxflags) $<
dep:
	(for f in $(srcs);do echo -n "";$(cxx) -E $(cxx-flags) -M -MM -MG $$f;done) >> .dep
clean:
	rm -rf *.o $(prj)
### Dependencies:
-include .dep