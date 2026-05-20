# Applied Sciences Overleaf Notes

## Files

- Main manuscript: [applsci_mdpi_overleaf.tex](C:/Users/Administrator/paper_数字媒体/manuscript/applsci_mdpi_overleaf.tex)
- Single-file fallback manuscript: [applsci_mdpi_overleaf_singlefile.tex](C:/Users/Administrator/paper_数字媒体/manuscript/applsci_mdpi_overleaf_singlefile.tex)
- Tables: [applsci_tables.tex](C:/Users/Administrator/paper_数字媒体/manuscript/applsci_tables.tex)
- References: [applsci_refs.bib](C:/Users/Administrator/paper_数字媒体/manuscript/applsci_refs.bib)
- Figure prompts: [applsci_figure_prompts.md](C:/Users/Administrator/paper_数字媒体/manuscript/applsci_figure_prompts.md)

## How to use in Overleaf

1. Start from the official MDPI article template in Overleaf.
2. Select the journal option for `Applied Sciences`.
3. Replace the sample `*.tex` content with `applsci_mdpi_overleaf.tex`.
4. Upload `applsci_tables.tex` and `applsci_refs.bib` into the same folder as the main `.tex` file.
5. Keep the template `Definitions/` folder unchanged.
6. Replace the two figure placeholders with final figures generated from the prompts in `applsci_figure_prompts.md`.

## If the template still fails to compile

Use the single-file version:

- [applsci_mdpi_overleaf_singlefile.tex](C:/Users/Administrator/paper_数字媒体/manuscript/applsci_mdpi_overleaf_singlefile.tex)

This file:

- removes unsupported MDPI metadata commands that may differ across template versions;
- inlines all tables;
- inlines the reference list;
- avoids dependency on `applsci_tables.tex` and `applsci_refs.bib`.

## Notes

- This manuscript is adapted to the MDPI / Applied Sciences template structure.
- The citation style has been converted to numeric `\cite{}` commands, which is the standard style used by the MDPI article template for this journal configuration.
- Author metadata, affiliations, editorial information, and correspondence details are placeholders and should be replaced before submission.
