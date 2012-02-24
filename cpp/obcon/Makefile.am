AM_CPPFLAGS = -I${TBLIB_ROOT}/include/tbsys \
			  -I${TBLIB_ROOT}/include/tbnet \
			  -I${top_srcdir}/include \
        -I${top_srcdir}/tests/updateserver \
			  -I${top_srcdir}/src \
				-fPIC

LDADD = $(top_builddir)/src/updateserver/libupdateserver.a \
	$(top_builddir)/src/rootserver/librootserver.a \
	$(top_builddir)/src/chunkserver/libchunkserver.a  \
	$(top_builddir)/src/mergeserver/libmergeserver.a \
	$(top_builddir)/src/sstable/libsstable.a \
	$(top_builddir)/src/common/libcommon.a \
		$(top_builddir)/src/common/compress/libcomp.a \
		$(top_builddir)/src/common/btree/libbtree.a \
		${TBLIB_ROOT}/lib/libtbnet.a \
		${TBLIB_ROOT}/lib/libtbsys.a

AM_LDFLAGS = -lpthread -lc -lm -lrt -ldl -laio -lnuma -lcrypt -lssl -lreadline -lncurses -lcrypt -lssl ${GCOV_LIB}
CXXFLAGS = -D__STDC_LIMIT_MACROS -g -Wall -DORDER_SERIALIZE -DCOMPATIBLE -D_BTREE_ENGINE_ #-O2 -fno-strict-aliasing
if COVERAGE
CXXFLAGS+=-fprofile-arcs -ftest-coverage
AM_LDFLAGS+=-lgcov
endif

bin_PROGRAMS = obcon
obcon_SOURCES = ob_console.cpp reg_utils.cpp utils.cpp
clean-local:
	-rm -f *.gcov *.gcno *.gcda
