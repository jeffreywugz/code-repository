include comm.makefile
$(eval $(call obj-copy,prj,project))
prj.name = foo
prj.version = 0.1
prj.top-dir = $(shell basename `pwd`)
prj.programs = foo1 foo2
prj.scripts = foo3

$(eval $(call obj-copy,program-template,program))
program-template.test-args = --test
program-template.lang = c
program-template.cc = gcc
program-template.cppflags =
program-template.cflags = $(program-template.cppflags) -Wall -g -c
program-template.ldflags =  

$(eval $(call obj-copy,foo1,program-template))
$(eval $(call obj-copy,foo2,program-template))

foo1.objs = foo1.o
foo1.exe = foo1

foo2.objs = foo2.o
foo2.private-objs = 
foo2.exe = foo2
foo2.cflags += 
foo2.ldflags += 
foo2.run-args += 
foo2.tmp-files += 

$(eval $(call obj-copy,script-template,script))
script-template.lang = py
script-template.run-dep = $(foo1.exe) $(foo2.exe)

$(eval $(call obj-copy,foo3,script-template))
foo3.exe = foo3.py
foo3.doc = just a script for testing makefile
foo3.run-arg = 

all: foo3.test

$(eval $(call project-setup-targets,prj))
exe run test dep clean bak commit: %: prj.%