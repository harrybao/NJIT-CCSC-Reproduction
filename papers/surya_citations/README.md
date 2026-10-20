# Citations for the tools on the IBM Surya blog

The blog post (https://research.ibm.com/blog/surya-heliophysics-ai-model-sun) has no Citation button. I followed each link on it (`blog_links.txt`) and recorded the citation that the linked resource itself asks for. The `.bib` files in `bibtex/` were copied by script from the fetched pages, without retyping, and `bibtex/SOURCES.tsv` gives the exact URL and line range for each. `citation_notes.md` has the full record, including what could not be reached. Its paths under `/home/user/work/...` refer to raw page copies that are not included here.

| Linked from the blog | What it asks you to cite | File |
|---|---|---|
| github.com/NASA-IMPACT/Surya (HEAD 78b827e; only tag v1.0.0) | the Surya paper, arXiv:2508.14112. README section "Citation", added after v1.0.0 | `surya_github_readme_HEAD-78b827e.bib` |
| huggingface.co/nasa-ibm-ai4science/Surya-1.0 | the Surya paper, but the card's entry still has the placeholder `eprint={XXXX.XXXXX}` and the 32 authors of arXiv v1 | `surya_hf_Surya-1.0_card.bib` |
| arXiv 2508.14112 (Surya, v2) | arXiv export, 33 authors | `surya_arxiv_2508.14112.bib` |
| arXiv 2508.14107 (SuryaBench, v1) and the SuryaBench datasets | arXiv export. The SuryaBench GitHub README gives a different author list, and each HF dataset has its own DOI (10.57967/hf/7273 to 7280) pinned to a commit. The published version is *Scientific Data* 13:712 (2026), doi:10.1038/s41597-026-06552-5 | `suryabench_*.bib`, `hf_doi_*.bib` |
| huggingface.co/collections/nasa-impact/suryabench-... | not reachable (HTTP 401); the same datasets are public under nasa-ibm-ai4science | – |
| github.com/IBM/terratorch, ibm.github.io/terratorch/stable/ | the TerraTorch paper, arXiv:2503.20563 (docs "Citation" page). The docs link on the blog now returns 404; the docs are at torchgeo.org/terratorch | `terratorch_docs_citation.bib`, `terratorch_arxiv_2503.20563.bib` |
| arXiv 2304.06446 (SpectFormer) | arXiv export; the project page and README give `patro2023spectformer` | `spectformer_*.bib` |
| arXiv 2107.02192 (Long-Short Transformer, NeurIPS 2021) | arXiv export | `long-short-transformer_arxiv_2107.02192.bib` |

The four fine-tuned Surya models on Hugging Face (flares, solar wind, EUV spectra, AR segmentation) ask for no citation. None of these repositories has a CITATION.cff, so GitHub shows no "Cite this repository" button for any of them. The AWS Open Data registry entry for the S3 bucket asks for a different, 2024 paper (arXiv:2410.10841, `aws_registry_cited_arxiv_2410.10841.bib`).
