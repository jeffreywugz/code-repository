obj.doc = base object
obj.targets = print help
obj.attributes = attributes doc targets

define obj-copy
$(foreach attr, $($(2).attributes),  $(eval $(1).$(attr) = $($(2).$(attr))))
endef

define obj-setup-targets
$(1).print:
	@echo description of $(1):
	@echo doc: $$($(1).doc)
	@echo targets: $$($(1).targets)
$(1).help:
	@echo targets: $$($(1).targets)
endef


$(eval $(call obj-copy,program,obj))
program.doc = program is linked by several obj file.
program.lang = c
program.objs =
program.private-objs =
program.extra-objs =
program.exe = a.out
program.run-args =
program.test-args = --test
program.cc = gcc
program.cpp = cpp
program.cflags = -Wall -Werror -g
program.cppflags =
program.ldflags =
program.tmp-files =
program.attributes += lang objs private extra-objs exe run-args test-args cc cpp cflags cppflags ldflags tmp-files

define program-setup-targets
$(eval $(call obj-setup-targets,$(1)))
$(1).targets += exe run test clean dep
$(1).expanded-private-objs = $$(patsubst %.o,%.$(1).private.o,$$($(1).private-objs))
$(1).exe: $$($(1).exe)
$(1).run: $$($(1).exe)
	./$$< $$($(1).run-args)
$(1).test: $$($(1).exe)
	./$$< $$($(1).test-args)
$(1).clean:
	rm -rf $$($(1).objs)  $$($(1).expanded-private-objs) $$($(1).exe) $$($(1).tmp-files) .$(1).dep
$$($(1).exe): $$($(1).objs) $$($(1).expanded-private-objs) $$($(1).extra-objs)
	$$($(1).cc) $$($(1).ldflags) -o $$@ $$^
$$($(1).objs): %.o: %.$($(1).lang)
	$$($(1).cc) $$($(1).cflags) -o $$@ $$<
$$($(1).expanded-private-objs): %.$(1).private.o: %.$($(1).lang)
	$$($(1).cc) $$($(1).cflags) -o $$@ $$<
$(1).dep:
	(for f in $$(patsubst %.o,%.$$($(1).lang),$$($(1).objs)); do echo -n "";$$($(1).cpp) $$($(1).cppflags) -M -MM -MG $$$$f;done) > .$(1).dep
-include .$(1).dep
endef

$(eval $(call obj-copy,script,obj))
script.doc = script need not compile
script.lang = py
script.exe = a.py
script.run-dep =
script.run-arg =
script.test-arg = --test
script.tmp-files =
script.attributes += lang exe run-dep run-arg test-arg tmp-files

define script-setup-targets
$(eval $(call obj-setup-targets,$(1)))
$(1).targets += run test clean
$(1).run: $$($(1).run-dep) $$($(1).exe)
	./$$($(1).exe) $$($(1).run-arg)
$(1).test: $$($(1).run-dep) $$($(1).exe)
	./$$($(1).exe) $$($(1).test-arg)
$(1).clean:
	rm -rf $$($(1).tmp-files) 
endef

$(eval $(call obj-copy,project,obj))
project.doc = a project can hold several programs.
project.name =
project.version = 0.1
project.top-dir =
project.programs =
project.scripts =
project.tmp-files =
script.attributes += doc name version top-dir programs scripts tmp-files

define project-setup-targets
$(eval $(call obj-setup-targets,$(1)))
$(foreach program, $($(1).programs), $(eval $(call program-setup-targets,$(program))))
$(foreach script, $($(1).scripts), $(eval $(call script-setup-targets,$(script))))
$(1).targets += exe run test dep clean bak commit
$(1).exe: $$(patsubst %,%.exe,$$($(1).programs))
$(1).run: $$(patsubst %,%.run,$$($(1).programs))
$(1).test: $$(patsubst %,%.test,$$($(1).programs))
$(1).dep: $$(patsubst %,%.dep,$$($(1).programs))
$(1).clean: $$(patsubst %,%.clean,$$($(1).programs))
	rm -rf $$($(1).tmp-files) 
$(1).bak: $(1).clean
	(cd ..; tar jcf $$($(1).name)-$$($(1).version).tar.bz2 $$($(1).top-dir); mv $$($(1).name)-$$($(1).version).tar.bz2 ~/arc)
$(1).commit: $(1).clean
	svn commit .
endef


