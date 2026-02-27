PDF=main.pdf
TEX=main.tex

.PHONY: pdf clean

pdf:
	latexmk -pdf $(TEX)

clean:
	latexmk -C $(TEX)
