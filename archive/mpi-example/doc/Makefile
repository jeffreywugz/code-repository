PRJ=doc
TEX=pdflatex
DOCTYPE=pdf
VIEWER=evince
.PHONY: all view clean

all: view
$(PRJ).$(DOCTYPE): $(PRJ).tex
	$(TEX) $^
view: $(PRJ).$(DOCTYPE)
	$(VIEWER) $^
clean:
	rm -rf *.dvi *.aux *.log *.toc *.out
