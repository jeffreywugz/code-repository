PRJ=wmutil
modules=$(shell find . -maxdepth 1 ! -name ".*" -type d)
.PHONY: all checkout commit bak

all: commit
clean:
	for dir in $(modules); do (cd $$dir; make clean) done
checkout:
	(cd ..; svn checkout https://gtk-dmenu.googlecode.com/svn/trunk/ wmutil --username huafengxi)
commit: clean
	svn commit  .
bak: clean
	(cd ..; tar jcf $(PRJ).tar.bz2 $(PRJ); mv $(PRJ).tar.bz2 ~/arc)
