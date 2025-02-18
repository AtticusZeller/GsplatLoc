

[Zotero Style Repository](https://www.zotero.org/styles) down your csl

## install tex tools
```powershell
winget install miktex
winget install JohnMacFarlane.Pandoc
# restart
```
## python env
```powershell
uv sync --all-extras --all-groups
```

##  Export
### PDF
```powershell
uv run  pandoc --from markdown --to pdf -o main.pdf --pdf-engine=xelatex main.md --mathjax --citeproc --template=template.tex --filter=table_filter.py --csl=ieee.csl --bibliography=bibliography.bib
```

### Tex
```powershell
uv run pandoc --from markdown --to latex -o main.tex --pdf-engine=xelatex main.md --mathjax --citeproc --template=template.tex --filter=table_filter.py --csl=ieee.csl --bibliography=bibliography.bib
```

```powershell
uv run pandoc --from markdown --to latex -o .\submission\main.tex main.md --template=output.tex --biblatex --filter=table_filter.py --bibliography=bibliography.bib --mathjax
```


```powershell
pdflatex main
biber main
pdflatex main
pdflatex main
```


## Clean
```powershell
rm *.aux,*.bbl,*.blg,*.log, *.bcf, *.out,*.run.xml -Force
```

## Zip

```powershell
zip arxiv-submission.zip main.tex arxiv.sty bibliography.bib Flowcharts.pdf main.bbl orcid.pdf
```
