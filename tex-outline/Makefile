tex = pdflatex --shell-escape
target = demo
demo.pdf: outline.sty demo.tex

all: view
view: $(target).pdf
	evince $<
%.pdf: %.tex
	$(tex) $<
clean:
	rm -f *.log *.aux *.out