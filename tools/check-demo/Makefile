include comm.makefile
$(eval $(call obj-copy,prj,project))
prj.name = foo
prj.version = 0.1
prj.top-dir = $(shell basename `pwd`)
prj.programs = check-demo

$(eval $(call obj-copy,program-template,program))
program-template.test-args = --test
program-template.lang = c
program-template.cc = gcc
program-template.cppflags = -I/usr/local/include
program-template.cflags = $(program-template.cppflags) -Wall -g -c
program-template.ldflags =  -L/usr/local/lib -lcheck
$(eval $(call obj-copy,check-demo,program-template))
check-demo.objs = check-demo.o
check-demo.exe = check-demo

all: check-demo.test

$(eval $(call project-setup-targets,prj))
exe run test dep clean bak commit: %: prj.%