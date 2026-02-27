PDF=main.pdf
TEX=main.tex

.PHONY: pdf pdf-numerisk pdf-hrp clean

pdf:
	latexmk -pdf $(TEX)

pdf-numerisk:
	latexmk -cd -pdf "Teori/Numerisk Ustabilitet.tex"

pdf-hrp:
	latexmk -cd -pdf "Teori/HRP Teori.tex"

clean:
	latexmk -C $(TEX)
