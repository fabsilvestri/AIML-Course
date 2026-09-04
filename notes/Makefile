# Part V extended lecture notes.
#
#     make            # build any PDF whose .tex or preamble is newer
#     make clean      # remove the intermediates, keep the PDFs
#     make distclean  # remove the PDFs too
#
# Two pdflatex passes, because the table of contents needs the first one's
# .toc file. latexmk would do this for us; two explicit passes are one fewer
# dependency and the build takes three seconds either way.

# every document here except the shared preamble, which is \input by them
TEX  := $(filter-out preamble.tex,$(wildcard *.tex))
PDFS := $(TEX:.tex=.pdf)

all: $(PDFS)

%.pdf: %.tex preamble.tex
	pdflatex -interaction=nonstopmode -halt-on-error $< > /dev/null
	pdflatex -interaction=nonstopmode -halt-on-error $< > /dev/null
	@echo "built $@ ($$(pdfinfo $@ | awk '/^Pages/{print $$2}') pages)"

clean:
	rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk

distclean: clean
	rm -f $(PDFS)

.PHONY: all clean distclean
