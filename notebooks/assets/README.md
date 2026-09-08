# Notebook diagrams

These files replace runtime `TikzPictures` rendering in `feynman.jl` and
`reversibleprog.jl`, so the notebooks display their diagrams without a local
LaTeX installation. The notebooks use the SVG files; the PDF files preserve the
original requested output format, and the matching `.tex` files are the sources.

The assets were generated with Tectonic 0.17.0 and converted to SVG with
`pdftocairo` 26.03.0:

```sh
tectonic --outdir notebooks/assets notebooks/assets/NAME.tex
pdftocairo -svg notebooks/assets/NAME.pdf notebooks/assets/NAME.svg
```

`reversibleprog-14.tex` is the expanded form of the 42-step diagram that the
notebook previously constructed with Julia string interpolation.
