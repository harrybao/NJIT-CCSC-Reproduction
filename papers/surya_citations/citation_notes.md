# Citations for every tool/resource linked from the IBM Surya blog

- Blog: https://research.ibm.com/blog/surya-heliophysics-ai-model-sun (saved copy `/home/user/work/surya_scout/blog.html`; JSON-LD `"datePublished":"2025-08-20T13:00:00.000Z"`, `"dateModified":"2025-08-25T15:30:44.540Z"`).
- Everything below was fetched with `curl` on 2026-10-15 (UTC). Raw copies of every page/API response are under `/home/user/work/surya_scout/research/sources/` (sub-folders `github/ hf/ doi/ arxiv/ docs/ zenodo/ aws/`).
- Every BibTeX block in this file is inserted by a script straight from the saved source file (no retyping). The same blocks are saved one per file in `/home/user/work/surya_scout/research/bibtex/`. `bibtex/SOURCES.tsv` maps each `.bib` file to its exact source (URL + line range) and a SHA-1 prefix. The only change: arXiv/Zenodo/DOI exports that had no trailing newline got one.
- The upstream clone `/home/user/upstream/surya/Surya` was only read (`git show/log/grep`). `git status` is clean.

## 0. Key findings

1. **The blog has no "Citation" button and no citation text.** A search of `blog.html` for `citation|cite|bibtex|doi.org` finds nothing. Every citation lives in the linked resources.
2. **Surya (the model and code)**: the resources ask for the Surya paper, arXiv:2508.14112. They give **three different BibTeX entries**:
   - GitHub README: `@article{roy2025surya, ... journal={arXiv preprint arXiv:2508.14112}}` with 10 authors + "others".
   - HF model card: `@misc{roy2025surya, ...}` with 32 authors, but its `eprint={XXXX.XXXXX}` and `url={https://arxiv.org/abs/XXXX.XXXXX}` are **unfilled placeholders**.
   - arXiv export: `@misc{roy2025suryafoundationmodelheliophysics, ...}` with 33 authors. arXiv **v2 added "Anne Spalding"**, which the HF card (it matches v1) does not have.
3. **SuryaBench (the datasets)**: none of the 7 HF dataset cards has a BibTeX entry for SuryaBench. The paper is arXiv:2508.14107. The SuryaBench GitHub README gives `@misc{suryabench}` with **21 authors**, while arXiv lists **25**. Each HF dataset has its own DataCite DOI (10.57967/hf/7273–7280, without 7277), and each DOI pins a commit. Zenodo has three SuryaBench records (code V1.0, a DOI-only "dataset" record that points to AWS, and a talk). The AWS Open Data Registry "How to Cite" names a **different, 2024 paper** (arXiv:2410.10841).
4. **TerraTorch**: there is no CITATION.cff. The docs "Citation" page asks for `@article{gomes2025terratorch, ... arXiv:2503.20563}`. The blog's docs link `https://ibm.github.io/terratorch/stable/` is **dead (HTTP 404)**. The docs now live at `https://torchgeo.org/terratorch/stable/`.
5. **SpectFormer / Long-Short Transformer**: there are arXiv export BibTeX entries. SpectFormer's project page and GitHub README also give their own `@article{patro2023spectformer}`. Long-Short Transformer was "Published at NeurIPS 2021" (arXiv comment), but its code README gives no BibTeX. I could not get the NeurIPS proceedings entry (DBLP blocked).
6. **No CITATION.cff exists** in the Surya repo (none on any branch in its history), in TerraTorch (404 at HEAD), or in SuryaBench (404). So GitHub's "Cite this repository" button does not appear for any of them.
7. **Not reachable:** the GitHub REST API (releases/tags) for NASA-IMPACT/Surya (HTTP 403 from the session proxy); the blog's SuryaBench collection link (HTTP 401, private or deleted, no Wayback snapshot); ibm.github.io TerraTorch docs (404); DBLP (bot check). Details are in §9.

## 1. Summary table

| Resource (as linked from blog) | What the resource itself asks you to cite | Where that request lives | Version / revision it points to | Licence | `.bib` file(s) |
|---|---|---|---|---|---|
| GitHub `NASA-IMPACT/Surya` | Surya paper, `@article{roy2025surya}` | README.md "📄 Citation" (lines 241–252 at HEAD) | git `main` HEAD `78b827e4c9e01343210d4c896d6e915c0e03003e` (2026-03-07). Block unchanged since `3d4d8dd` (2025-09-12). **Not present at tag `v1.0.0`** (`bea20c12f77a28fafa89c892072aec890bf7ec09`, 2025-08-17). Points to arXiv 2508.14112 without a version. | Apache-2.0 (LICENSE, README) | `surya_github_readme_HEAD-78b827e.bib` (+ superseded `surya_github_readme_cff9f51-superseded.bib`) |
| HF model `nasa-ibm-ai4science/Surya-1.0` | Surya paper, `@misc{roy2025surya}` with placeholder eprint | Model card "## Citation" (README.md lines 84–97) | HF sha `5cc4b5386d5f78fda3896b1389589d4e173bf212` (lastModified 2025-08-20T00:57:57.000Z). No arXiv id (placeholder). | apache-2.0 (card YAML) | `surya_hf_Surya-1.0_card.bib` |
| Other Surya HF models (4 fine-tuned) | nothing | cards have no citation section | see §3.2 | apache-2.0 | – |
| HF datasets for Surya (8) | no BibTeX in any card. HF DOI per SuryaBench dataset. euv-spectra cites EVE instrument papers (plain text). | dataset cards; DataCite | see §3.3 (DOI `version` = commit) | cc-by-4.0 (7 SuryaBench sets), mit (`Surya-1.0_validation_data`) | `hf_doi_*.bib` (7) |
| HF collection `nasa-impact/suryabench-68265ce306fc2470c121af7b` | **unreachable (HTTP 401)** | – | – | – | – |
| ↳ public replacement `nasa-ibm-ai4science/suryabench-689276db205c56e81c10070c` | collections carry no citation; see items above + SuryaBench paper | – | lastUpdated 2025-12-15T19:15:24.734Z, 7 items | – | – |
| GitHub `IBM/terratorch` (now `torchgeo/terratorch`) | TerraTorch paper, `@article{gomes2025terratorch}` | docs page "Citation" (`docs/about/citation.md`). README only links the arXiv paper. | docs "stable" = 1.2.13. citation.md identical at tag `v1.2.13` (`d1267fbd…`), `main` (`f967ef77…`), branch `docs` (`b315bd90…`). arXiv 2503.20563 is v1 only. | Apache-2.0 AND MIT (pyproject) | `terratorch_docs_citation.bib`, `terratorch_arxiv_2503.20563.bib` |
| `https://ibm.github.io/terratorch/stable/` | **dead (HTTP 404)** | – | – | – | – |
| arXiv 2508.14112 (Surya) | arXiv export | https://arxiv.org/bibtex/2508.14112 | latest = **v2** (2025-08-21). Export has no version field. | CC BY 4.0 | `surya_arxiv_2508.14112.bib` |
| arXiv 2508.14107 (SuryaBench) | arXiv export; GitHub README `@misc{suryabench}` | arXiv; `NASA-IMPACT/SuryaBench` README "## Citation" | **v1** only (2025-08-18). Code tag `V1.0` = `ce8b841…` | CC BY 4.0 (paper); MIT (code) | `suryabench_arxiv_2508.14107.bib`, `suryabench_github_readme_V1.0.bib`, `suryabench_zenodo_*.bib` |
| arXiv 2304.06446 (SpectFormer) | arXiv export; project page/GitHub `@article{patro2023spectformer}` | arXiv; https://badripatro.github.io/SpectFormers/ "Citation"; `badripatro/SpectFormers` README "## Citation" | latest = **v2** (2023-04-14) | CC BY-NC-SA 4.0 (paper); no LICENSE file found in code repo | `spectformer_arxiv_2304.06446.bib`, `spectformer_project_page.bib`, `spectformer_github_readme.bib` |
| arXiv 2107.02192 (Long-Short Transformer; blog links the PDF) | arXiv export only | https://arxiv.org/bibtex/2107.02192 | latest = **v3** (2021-12-07). "Published at NeurIPS 2021" | CC BY-NC-SA 4.0 (paper); MIT (NVIDIA/transformer-ls code) | `long-short-transformer_arxiv_2107.02192.bib` |
| HF org, non-Surya members (Lunar FM, SomBench, 1 Space) | Lunar FM + SomBench BibTeX (+ source catalogues) | model/dataset cards "## Citation(s)" | see §3.4 | apache-2.0 (models, Space), cc-by-4.0 (datasets) | `lunar_hf_*.bib` (10) |

---

## 2. Surya GitHub repository — https://github.com/NASA-IMPACT/Surya

### 2.1 README citation (current)
Source: `/home/user/upstream/surya/Surya/README.md`, section `## 📄 Citation` (line 241). The intro sentence is on line 243:

```text
If you use Surya in your research, please cite our paper:
```

BibTeX (README lines 246–251; also identical to `https://raw.githubusercontent.com/NASA-IMPACT/Surya/main/README.md`, which I fetched and diffed: same file):

```bibtex
@article{roy2025surya,
  title={Surya: Foundation Model for Heliophysics},
  author={Roy, Sujit and Schmude, Johannes and Lal, Rohit and Gaur, Vishal and Freitag, Marcus and Kuehnert, Julian and van Kessel, Theodore and Hegde, Dinesha V and Mu{\~n}oz-Jaramillo, Andr{\'e}s and Jakubik, Johannes and others},
  journal={arXiv preprint arXiv:2508.14112},
  year={2025}
}
```
<sub>file: `bibtex/surya_github_readme_HEAD-78b827e.bib`</sub>

### 2.2 History of that citation block (from `git log -G` on README.md)
- Tag `v1.0.0` (commit `bea20c12f77a28fafa89c892072aec890bf7ec09`, 2025-08-17, "Merge pull request #11 from NASA-IMPACT/euv_spectra"): the README has **no Citation, BibTeX, arXiv or licence text** (grep is empty). So **the only tagged version of the code does not ask for any citation.**
- Commit `cff9f51fdc614e17de61d23bc013cb667155091b` (Rohit Lal, 2025-08-18, "update readme") added this first, now superseded entry:

```bibtex
@article{roy2024surya,
  title={Surya: Foundation Model for Heliophysics},
  author={Roy, Sujit and Schmude, Johannes and Lal, Rohit and Gaur, Vishal and Freitag, Marcus and others},
  journal={arXiv preprint},
  year={2025}
}
```
<sub>file: `bibtex/surya_github_readme_cff9f51-superseded.bib`</sub>

- Commit `3d4d8dd66879197ce81822ca5425da3ed3fd331d` (Rohit Lal, 2025-09-12, "Corrected BibTex and added Paper Link") replaced it with the current entry (§2.1). The block is byte-identical at `a9724c7`, `3e868c1` and HEAD `78b827e`.

### 2.3 CITATION.cff / Zenodo
- `https://raw.githubusercontent.com/NASA-IMPACT/Surya/main/CITATION.cff` gives **404**. `git log --all -- CITATION.cff '*.cff'` finds **0 commits** on all 9 remote branches (the clone is not shallow, 92 commits on main). So there is no GitHub "Cite this repository" widget.
- Zenodo search (`https://zenodo.org/api/records?q=...`): `"NASA-IMPACT/Surya"` gives 0 hits. `"Surya: Foundation Model for Heliophysics"` gives 0 hits. So **I found no Zenodo DOI** for the Surya code (the search is keyword-based).

### 2.4 Tags / releases / versions
- `git ls-remote --tags https://github.com/NASA-IMPACT/Surya.git` returns exactly one tag: `bea20c12f77a28fafa89c892072aec890bf7ec09 refs/tags/v1.0.0`. It is lightweight (no `^{}` line, `git cat-file -t v1.0.0` = commit).
- `main`/HEAD = `78b827e4c9e01343210d4c896d6e915c0e03003e` (2026-03-07 "Merge pull request #46 from NASA-IMPACT/rohit/zs_inference"). `git describe --tags` = `v1.0.0-65-g78b827e`, so HEAD is 65 commits past the tag.
- `pyproject.toml` declares `version = "0.1.0"`, which differs from the git tag `v1.0.0`. The HF weights file is `surya.366m.v1.pt`.
- **GitHub Releases: NOT REACHABLE.** `https://api.github.com/repos/NASA-IMPACT/Surya/releases`, `/tags`, the repo endpoint and the HTML `https://github.com/NASA-IMPACT/Surya/releases` / `/tags` all returned **HTTP 403** from the session's egress proxy (`"GitHub access to this repository is not enabled for this session..."`). `add_repo` (read) confirmed that only anonymous git reads are available. So release titles, notes and assets are **unknown**. Only the git tag list above is confirmed.

### 2.5 Licence
`LICENSE` = "Apache License Version 2.0, January 2004". README `## 📜 License` (line 256):

```text
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
```

---

## 3. Hugging Face org `nasa-ibm-ai4science`

Listings: `https://huggingface.co/api/models?author=nasa-ibm-ai4science` gives **9 models**. `/api/datasets?author=...` gives **13 datasets**. `/api/spaces?author=...` gives 1 Space. `/api/collections?owner=nasa-ibm-ai4science` gives 4 collections. For every repo I saved `/api/<type>/<id>` (`hf/<type>/*.api.json`), the raw card `raw/main/README.md` and `tree/main`. The "sha" below is the `sha` field of `/api/<type>/<id>` at fetch time. For Surya-1.0 I also fetched `raw/<sha>/README.md` and it is byte-identical to `raw/main`. No API record has a `citation` field.

### 3.1 Model `nasa-ibm-ai4science/Surya-1.0`
- sha `5cc4b5386d5f78fda3896b1389589d4e173bf212`, createdAt 2025-08-13T15:22:00.000Z, lastModified 2025-08-20T00:57:57.000Z (latest commit "Upload SuryaFM.pdf"). Licence `apache-2.0` (card YAML). The API has no `arxiv:` tag.
- Files (tree/main): `surya.366m.v1.pt` 1800384309 bytes, `SuryaFM.pdf` 18347597 bytes, `config.yaml` 680, `scalers.yaml` 3835, `surya_arch.png`, `README.md` 4206. I downloaded none of them.
- Card `## Citation` (line 84). Intro (line 86):

```text
If you use Surya in your research, please cite:
```

BibTeX (lines 89–97). **Note: the card's ```` ```bibtex ```` fence is never closed, and the arXiv id is the placeholder `XXXX.XXXXX`**:

```bibtex
@misc{roy2025surya,
      title={Surya: Foundation Model for Heliophysics}, 
      author={Sujit Roy and Johannes Schmude and Rohit Lal and Vishal Gaur and Marcus Freitag and Julian Kuehnert and Theodore van Kessel and Dinesha V. Hegde and Andrés Muñoz-Jaramillo and Johannes Jakubik and Etienne Vos and Kshitiz Mandal and Ata Akbari Asanjan and Joao Lucas de Sousa Almeida and Amy Lin and Talwinder Singh and Kang Yang and Chetraj Pandey and Jinsu Hong and Berkay Aydin and Thorsten Kurth and Ryan McGranaghan and Spiridon Kasapis and Vishal Upendran and Shah Bahauddin and Daniel da Silva and Nikolai V. Pogorelov and Campbell Watson and Manil Maskey and Madhulika Guhathakurta and Juan Bernabe-Moreno and Rahul Ramachandran},
      year={2025},
      eprint={XXXX.XXXXX},
      archivePrefix={arXiv},
      primaryClass={astro-ph.SR},
      url={https://arxiv.org/abs/XXXX.XXXXX}, 
}
```
<sub>file: `bibtex/surya_hf_Surya-1.0_card.bib`</sub>

- Its 32 authors match arXiv **v1** exactly, in the same order. arXiv v2 adds "Spalding, Anne" between Pogorelov and Watson (checked by script against the `citation_author` meta tags of `arxiv.org/abs/2508.14112v1` and the v2 export).

### 3.2 Fine-tuned Surya models (no citation requested)
| repo | sha | lastModified | licence | card content relevant to citation |
|---|---|---|---|---|
| `solar_flares_surya` | `ec7c42ab22fbf126c4bfffadb420f6e45a669832` | 2025-08-18T22:22:30.000Z | apache-2.0 | card is only the YAML header `license: apache-2.0` (31 bytes) |
| `solar_wind_surya` | `6ac4aa79a96f8785a1ab300842d13727e52b58a5` | 2025-08-18T22:37:24.000Z | apache-2.0 | same (31 bytes) |
| `euv_spectra_surya` | `6ed3bc13eeece973653422cc83e583aabd31dae0` | 2025-08-26T15:42:52.000Z | apache-2.0 | same (31 bytes) |
| `ar_segmentation_surya` | `73a6de53c4bea812f7bc2ac29b9351ae2f2fe5fe` | 2025-08-20T20:20:32.000Z | apache-2.0 | `base_model: nasa-ibm-ai4science/Surya-1.0`. Results table and "Acknowledgements" ("**NASA IMPACT** and **IBM** for developing the Surya foundation model"). No citation. |

### 3.3 Surya-related datasets
None of these cards has a BibTeX block. Seven have a DataCite DOI (HF tag `doi:`). DataCite metadata comes from `https://api.datacite.org/dois/<doi>`. BibTeX comes from DOI content negotiation (`curl -L -H "Accept: application/x-bibtex" https://doi.org/<doi>`, served by data.crosscite.org). In every case DataCite `creators` = "NASA-IBM AI4Science", publisher "Hugging Face", year 2025, and `rightsList` is empty. **The DataCite `version` field is the HF commit the DOI was minted on, and it equals the current HF sha in all 7 cases.**

| dataset | HF sha (current) | lastModified | licence | DOI (DataCite `version`) | earlier DOI (version) | authors/contact named in card |
|---|---|---|---|---|---|---|
| `core-sdo` | `9c10abe52c1f206d1b66715745229e29ef543cc6` | 2025-12-16T21:21:41.000Z | cc-by-4.0 | 10.57967/hf/7273 (`9c10abe`) | 10.57967/hf/7271 (`cf284e2`) | "Sujit Roy, Dinesha V Hegde, Johannes Schmude, Rohit Lal, Vishal Gaur" |
| `surya-bench-flare-forecasting` | `bf474bc1e7752bd529e9707650f9310db8693763` | 2025-12-16T21:25:30.000Z | cc-by-4.0 | 10.57967/hf/7274 (`bf474bc`) | 10.57967/hf/7138 (`34f0d3d`) | Jinsu Hong, Kang Yang, Berkay Aydin |
| `surya-bench-coronal-extrapolation` | `622892941688f21e69bbad37598b33787083cdbd` | 2025-12-16T21:28:30.000Z | cc-by-4.0 | 10.57967/hf/7275 (`6228929`) | 10.57967/hf/7137 (`cfa383d`) | Contact: Daniel da Silva |
| `Surya-bench-solarwind` | `20547a25cbd3aaa359e9cda18be913b94c225e7b` | 2025-12-16T21:30:54.000Z | cc-by-4.0 | 10.57967/hf/7276 (`20547a2`) | 10.57967/hf/7139 (`23dd361`) | Vishal Upendran |
| `surya-bench-ar-segmentation` | `615a7df4ac3498c97d0d039e7b0654774ad86602` | 2025-12-16T21:48:14.000Z | cc-by-4.0 | 10.57967/hf/7278 (`615a7df`) | 10.57967/hf/7140 (`c7cf6b3`) | Jinsu Hong, Kang Yang, Berkay Aydin |
| `ar_emergence` | `48970a12c290b515912baadba87fa74a49e3c680` | 2025-12-16T21:51:53.000Z | cc-by-4.0 | 10.57967/hf/7279 (`48970a1`) | 10.57967/hf/7141 (`91f0b9c`) | Contact: Spyros Kasapis |
| `euv-spectra` | `aafccf9c1364bd39eaf49e058e39f24f70719657` | 2025-12-16T21:59:32.000Z | cc-by-4.0 | 10.57967/hf/7280 (`aafccf9`) | 10.57967/hf/7142 (`4a71eb7`) | Shah Mohammad Bahauddin |
| `Surya-1.0_validation_data` | `3b6710423d8ca10889e7d9f154f587b91266f6be` | 2025-08-17T17:22:41.000Z | **mit** | none | – | none (card: "Validation data for Surya 1.0") |

DOI BibTeX (verbatim, one file each):

```bibtex
@misc{https://doi.org/10.57967/hf/7273,
  doi = {10.57967/HF/7273},
  url = {https://huggingface.co/datasets/nasa-ibm-ai4science/core-sdo},
  author = {{NASA-IBM AI4Science}},
  title = {core-sdo},
  publisher = {Hugging Face},
  year = {2025}
}
```
<sub>file: `bibtex/hf_doi_core-sdo.bib`</sub>
```bibtex
@misc{https://doi.org/10.57967/hf/7274,
  doi = {10.57967/HF/7274},
  url = {https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-flare-forecasting},
  author = {{NASA-IBM AI4Science}},
  title = {surya-bench-flare-forecasting},
  publisher = {Hugging Face},
  year = {2025}
}
```
<sub>file: `bibtex/hf_doi_surya-bench-flare-forecasting.bib`</sub>
```bibtex
@misc{https://doi.org/10.57967/hf/7275,
  doi = {10.57967/HF/7275},
  url = {https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-coronal-extrapolation},
  author = {{NASA-IBM AI4Science}},
  title = {surya-bench-coronal-extrapolation},
  publisher = {Hugging Face},
  year = {2025}
}
```
<sub>file: `bibtex/hf_doi_surya-bench-coronal-extrapolation.bib`</sub>
```bibtex
@misc{https://doi.org/10.57967/hf/7276,
  doi = {10.57967/HF/7276},
  url = {https://huggingface.co/datasets/nasa-ibm-ai4science/Surya-bench-solarwind},
  author = {{NASA-IBM AI4Science}},
  title = {Surya-bench-solarwind},
  publisher = {Hugging Face},
  year = {2025}
}
```
<sub>file: `bibtex/hf_doi_Surya-bench-solarwind.bib`</sub>
```bibtex
@misc{https://doi.org/10.57967/hf/7278,
  doi = {10.57967/HF/7278},
  url = {https://huggingface.co/datasets/nasa-ibm-ai4science/surya-bench-ar-segmentation},
  author = {{NASA-IBM AI4Science}},
  title = {surya-bench-ar-segmentation},
  publisher = {Hugging Face},
  year = {2025}
}
```
<sub>file: `bibtex/hf_doi_surya-bench-ar-segmentation.bib`</sub>
```bibtex
@misc{https://doi.org/10.57967/hf/7279,
  doi = {10.57967/HF/7279},
  url = {https://huggingface.co/datasets/nasa-ibm-ai4science/ar_emergence},
  author = {{NASA-IBM AI4Science}},
  title = {ar_emergence},
  publisher = {Hugging Face},
  year = {2025}
}
```
<sub>file: `bibtex/hf_doi_ar_emergence.bib`</sub>
```bibtex
@misc{https://doi.org/10.57967/hf/7280,
  doi = {10.57967/HF/7280},
  url = {https://huggingface.co/datasets/nasa-ibm-ai4science/euv-spectra},
  author = {{NASA-IBM AI4Science}},
  title = {euv-spectra},
  publisher = {Hugging Face},
  year = {2025}
}
```
<sub>file: `bibtex/hf_doi_euv-spectra.bib`</sub>

Plain-text citations inside cards (verbatim):
- `euv-spectra` card, `## Citation` (line 80), lines 82–84 (instrument/data-product references, not a SuryaBench citation):

```text
Woods, T. N., Eparvier, F. G., Hock, R., Jones, A. R., Woodraska, D., Judge, D., ... & Viereck, R. (2012). Extreme Ultraviolet Variability Experiment (EVE) on the Solar Dynamics Observatory (SDO): Overview of science objectives, instrument design, data products, and model developments. The solar dynamics observatory, 115-143, doi: http://doi.org/10.1007/s11207-009-9487-6

Woodraska, D., & Eparvier, F. G. (2024). Solar Dynamics Observatory (SDO) Extreme Ultraviolet Variability Experiment (EVE): Version 8 science data product Release Notes,  LASP / University of Colorado Boulder Technical Document. https://lasp.colorado.edu/eve/data_access/eve_data/products/level2b/EVE_L2B_V8_README.pdf
```

- `ar_emergence` card, `## References` (lines 117–118). This is the work the data was used in, not a formal "please cite":

```text
## References
[1] Kasapis, S., Kitiashvili, I. N., Kosovichev, A. G. & Stefan, J. T. Prediction of intensity variations associated with emerging active regions using helioseismic power maps and machine learning. The Astrophys. J. Suppl. Ser. 10.3847/1538-4365/adfbe2 (2025)
```

- `core-sdo` card: no citation. It sends users to the AWS Open Data Registry (`https://registry.opendata.aws/surya-bench/`, see §5.3) and to `https://huggingface.co/collections/nasa-ibm-ai4science/suryabench` (HTTP 200).

### 3.4 Other org members (not Surya; listed because the task covers the whole org)
| repo | type | sha | lastModified | licence | citation in card |
|---|---|---|---|---|---|
| `NASA-IBM-Lunar-Foundation-Model` | model | `b657bdfe6fc4c0445d918f77a2d7b919805f3bce` | 2026-09-23T09:12:05.000Z | apache-2.0 | `## Citations` (line 205): `fraccaro2026lfm` + `sombench2026collection`. API tags `arxiv:2504.11171`, `arxiv:2609.13283`. |
| `Crater-Detection-NASA-IBM-Lunar-Foundation-Model` | model | `f09ebeae2099c28f09bf4060f56d3e327a6192de` | 2026-09-09T17:23:25.000Z | apache-2.0 | `## Citation` (line 275): above two + `Robbins2019New` |
| `IMP-Segmentation-NASA-IBM-Lunar-Foundation-Model` | model | `10a09ffae35ef5712c40c2a1fc41ec82d4881ea6` | 2026-09-09T17:15:45.000Z | apache-2.0 | `## Citation` (line 209): above two + `hargitai2025clusters` |
| `Ice-Prospectivity-NASA-IBM-Lunar-Foundation-Model` | model | `fb0510d0d676ae7bfae9acd88219b05449c769ee` | 2026-09-09T17:25:51.000Z | apache-2.0 | `## Citation` (line 255): above two + `coyan2025prospectivity` |
| `Sombench-NAC-Crater-Detection` | dataset | `79419eb3486b2dd9ad09a283802e0bb8d75c767d` | 2026-09-09T16:13:45.000Z | cc-by-4.0 | `## Citation` (line 111) |
| `Sombench-WAC-Crater-Detection` | dataset | `20f800becb64a0c7ad314652a683f97f0281fe67` | 2026-09-09T16:09:32.000Z | cc-by-4.0 | `## Citation` (line 81) + "Data sources" `robbins2019craters` |
| `Sombench-IMP-Segmentation` | dataset | `1418851011f4b1d4fd8140a8d765b17fda86ebc0` | 2026-09-09T16:15:44.000Z | cc-by-4.0 | `## Citation` (line 73) |
| `Sombench-Ice-Prospectivity-Regression` | dataset | `78040503ceaa68107829f48114774112e8ee940d` | 2026-09-09T16:17:50.000Z | cc-by-4.0 | `## Citation` (line 96) |
| `Sombench-pretraining-data` | dataset | `e52a7f46090458227f970e7ea19896c179b6c371` | 2026-09-09T16:49:03.000Z | cc-by-4.0 | `## Citations` (line 262) |
| Space `nasa-ibm-lunar-foundation-model-generation-demo` | space | `7d2e72ecd7bc0b08546ba738ed683856c67fa4a7` | 2026-09-12T20:31:08.000Z | apache-2.0 | none |

Each card's whole citation code block is saved verbatim as `bibtex/lunar_hf_<repo>.bib`. Differences between cards, found by hashing every entry:
- In the 5 SomBench **dataset** cards, `fraccaro2026lfm` has an extra `howpublished` line and **no comma after `year   = {2026}`**, so that entry **will not parse in BibTeX**. In the 4 **model** cards, `fraccaro2026lfm` has no `howpublished` line and parses.
- `fraccaro2026lfm` gives no venue or arXiv id, even though the model's API tags include `arxiv:2609.13283`.
- `sombench2026collection` uses two whitespace layouts (main LFM, Crater and Ice model cards vs IMP model card and all dataset cards). The text is otherwise the same.

Example (main Lunar FM card, lines 208–229):

```bibtex
@article{fraccaro2026lfm,
  title  = {Multimodal-Multiresolution Foundation Model for Lunar Remote Sensing},
  author = {Fraccaro, Paolo and Nyirjesy, Gabby and Szwarcman, Daniela and Patil, Himanshu
            and Gaur, Vishal and Lal, Rohit and Slank, Rachel A. and Dawson, Geoffrey
            and Debary, Hiyam and Dionelis, Nikolaos and Barker, Michael K. and Annex, Andrew
            and Viswanathan, Vishnu and Morse, Zachary and Schaefer, Ethan I. and Kumar, Ankur
            and Watson, Campbell D. and Dawson-Rigas, Rebekah I. and Maskey, Manil
            and Roy, Sujit and Ramachandran, Rahul and Bernab\'e-Moreno, Juan},
  year   = {2026}
}

@misc{sombench2026collection,
  author      = {Patil, Himanshu and Nyirjesy, Gabby and Slank, Rachel A. and Gaur, Vishal
                  and Szwarcman, Daniela and Fraccaro, Paolo and Dionelis, Nikolaos and Barker, Michael K.
                  and Annex, Andrew and Viswanathan, Vishnu and Morse, Zachary and Schaefer, Ethan I.
                  and Debary, Hiyam and Kumar, Ankur and Lal, Rohit and Dawson, Geoffrey
                  and Watson, Campbell and Dawson-Rigas, Rebekah I. and Maskey, Manil
                  and Bernab\'e-Moreno, Juan and Ramachandran, Rahul and Roy, Sujit},
  title        = {{SomBench}: Benchmark Dataset for Advancing Machine Learning in Lunar Science},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/collections/nasa-ibm-ai4science/lunar-fm-ml-ready-benchmark-dataset-sombench}}
}
```
<sub>file: `bibtex/lunar_hf_NASA-IBM-Lunar-Foundation-Model.bib`</sub>

Example of the dataset-card variant (`Sombench-pretraining-data`, lines 265–287; identical in all 5 SomBench dataset cards):

```bibtex
@article{fraccaro2026lfm,
  title  = {Multimodal-Multiresolution Foundation Model for Lunar Remote Sensing},
  author = {Fraccaro, Paolo and Nyirjesy, Gabby and Szwarcman, Daniela and Patil, Himanshu
            and Gaur, Vishal and Lal, Rohit and Slank, Rachel A. and Dawson, Geoffrey
            and Debary, Hiyam and Dionelis, Nikolaos and Barker, Michael K. and Annex, Andrew
            and Viswanathan, Vishnu and Morse, Zachary and Schaefer, Ethan I. and Kumar, Ankur
            and Watson, Campbell D. and Dawson-Rigas, Rebekah I. and Maskey, Manil
            and Roy, Sujit and Ramachandran, Rahul and Bernab\'e-Moreno, Juan},
  year   = {2026}
  howpublished = {\url{https://huggingface.co/collections/nasa-ibm-ai4science/nasa-ibm-lunar-fm-and-downstream-models}}
}

@misc{sombench2026collection,
  author = {Patil, Himanshu and Nyirjesy, Gabby and Slank, Rachel A. and Gaur, Vishal
          and Szwarcman, Daniela and Fraccaro, Paolo and Dionelis, Nikolaos and Barker, Michael K.
          and Annex, Andrew and Viswanathan, Vishnu and Morse, Zachary and Schaefer, Ethan I.
          and Debary, Hiyam and Kumar, Ankur and Lal, Rohit and Dawson, Geoffrey
          and Watson, Campbell and Dawson-Rigas, Rebekah I. and Maskey, Manil
          and Bernab\'e-Moreno, Juan and Ramachandran, Rahul and Roy, Sujit},
  title        = {{SomBench}: Benchmark Dataset for Advancing Machine Learning in Lunar Science},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/collections/nasa-ibm-ai4science/lunar-fm-ml-ready-benchmark-dataset-sombench}}
}
```
<sub>file: `bibtex/lunar_hf_Sombench-pretraining-data.bib`</sub>

Extra "please also cite" text (verbatim):

```text
Please also cite TerraMind (Jakubik et al., 2025), TerraTorch (Gomes et al., 2025), FlexiViT (Beyer et al., 2023),
and — for the NAC annotations — OpenCraterTool (Heyer et al., 2023).
```
```text
Please also cite TerraMind (Jakubik et al., 2025), TerraTorch (Gomes et al., 2025), and FlexiViT (Beyer et al.,
2023).
```
```text
Please also cite TerraMind (Jakubik et al., 2025), TerraTorch (Gomes et al., 2025), and FlexiViT (Beyer et al.,
2023). Data products courtesy of LOLA and Diviner.
```

### 3.5 Collections of the org
`nasa-ibm-ai4science/suryabench-689276db205c56e81c10070c` ("SuryaBench", lastUpdated 2025-12-15T19:15:24.734Z, 7 dataset items, see §4). `nasa-ibm-ai4science/surya-6a957f7eeebf28997ebf3b70` ("Surya", **0 items**, lastUpdated 2026-08-31T13:20:09.346Z). Plus 2 lunar collections. HF collections have no citation field.

---

## 4. SuryaBench collection (blog link)
- Blog link `https://huggingface.co/collections/nasa-impact/suryabench-68265ce306fc2470c121af7b`: **HTTP 401** for both `/api/collections/...` (`{"error":"Invalid username or password."}`) and the HTML page. It is not among the 5 public `nasa-impact` collections (`/api/collections?owner=nasa-impact`). `/api/collections/68265ce306fc2470c121af7b` gives 404. The Wayback Machine has no snapshot (`archived_snapshots: {}`). So it is **private or deleted. I could not read its items or any citation.**
- The SuryaBench paper's "6 Code Availability" (arXiv 2508.14107v1 HTML) uses the same dead link. The SuryaBench GitHub README links `https://huggingface.co/collections/nasa-impact/suryabench`, also **401**.
- Working public collection: `nasa-ibm-ai4science/suryabench-689276db205c56e81c10070c` (title "SuryaBench", description "Benchmark Dataset for Advancing Machine Learning in Heliophysics and Space Weather Prediction"). Items: `Surya-bench-solarwind`, `surya-bench-flare-forecasting`, `surya-bench-ar-segmentation`, `euv-spectra`, `surya-bench-coronal-extrapolation`, `ar_emergence`, `core-sdo` (all `nasa-ibm-ai4science/`). Their citations and versions are in §3.3. The short URL `https://huggingface.co/collections/nasa-ibm-ai4science/suryabench` returns 200.

---

## 5. SuryaBench paper's own code, Zenodo and AWS records (reached from the SuryaBench paper and the core-sdo card)

### 5.1 GitHub `NASA-IMPACT/SuryaBench` (linked in arXiv 2508.14107 "Code Availability")
- `git ls-remote`: HEAD = `ce8b841bb999a7a917bcc004cd17a57ca2109b81` = `refs/tags/V1.0` (the only tag). No CITATION.cff (404). LICENSE: "MIT License … Copyright (c) 2025 Inter Agency Implementation and Advanced Concepts".
- README `## Citation` / `**BibTeX:**` (lines 50–51), block lines 53–77:

```bibtex
@misc{suryabench,
      title={SuryaBench: Benchmark Dataset for Advancing Machine Learning in Heliophysics and Space Weather Prediction}, 
      author={ Sujit Roy and 
               Dinesha Vasanta Hegde and
               Johannes Schmude and
               Amy Lin and
               Vishal Gaur and
               Talwinder Singh and
               Rohit Lal and 
               Andrés Muñoz-Jaramillo and 
               Kang Yang and 
               Chetraj Pandey and
               Jinsu Hong and 
               Berkay Aydin and 
               Ryan McGranaghan and 
               Spiridon Kasapis and 
               Vishal Upendran and 
               Shah Bahauddin and 
               Daniel da Silva and 
               Marcus Freitag and 
               Nikolai Pogorelov and
               Manil Maskey and 
               Rahul Ramachandran},
      year={2025}
}
```
<sub>file: `bibtex/suryabench_github_readme_V1.0.bib`</sub>

  Compared with arXiv 2508.14107 (25 authors), this list (21) **omits Kshitiz Mandal, Iksha Gurung, Campbell Watson and Juan Bernabe-Moreno**, and puts Talwinder Singh **before** Rohit Lal (arXiv has Lal, Mandal, Singh).

### 5.2 Zenodo
- **Software, code V1.0**: DOI 10.5281/zenodo.17842820 (concept DOI 10.5281/zenodo.17842819). Title "NASA-IMPACT/SuryaBench: Suryabench Dataset", published 2025-12-06, version `V1.0`, licence `mit-license`, `isSupplementTo` `https://github.com/NASA-IMPACT/SuryaBench/tree/V1.0`. File `NASA-IMPACT/SuryaBench-V1.0.zip` is 1270904 bytes. The SWHID path ends in `NASA-IMPACT-SuryaBench-ce8b841`, which matches tag V1.0. Export `https://zenodo.org/records/17842820/export/bibtex`:

```bibtex
@software{jinsu_hong_2025_17842820,
  author       = {Jinsu Hong and
                  Lelouch vi' Britania and
                  Rohit Lal and
                  Dinesha Hegde and
                  Spyros Kasapis and
                  yoko3new and
                  Vishal and
                  Berkay Aydin and
                  shah-astro},
  title        = {NASA-IMPACT/SuryaBench: Suryabench Dataset},
  month        = dec,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {V1.0},
  doi          = {10.5281/zenodo.17842820},
  url          = {https://doi.org/10.5281/zenodo.17842820},
  swhid        = {swh:1:dir:38dc00f401da046b93ea69d6a27a1d7467405a77
                   ;origin=https://doi.org/10.5281/zenodo.17842819;vi
                   sit=swh:1:snp:e42967eeedacdc2301f00ce75ed61a9f3fca
                   4144;anchor=swh:1:rel:b1117435eec37825484760f020ea
                   a3bcaf95636e;path=NASA-IMPACT-SuryaBench-ce8b841
                  },
}
```
<sub>file: `bibtex/suryabench_zenodo_software_17842820.bib`</sub>

- **Dataset DOI record**: DOI 10.5281/zenodo.17872179 (concept 10.5281/zenodo.17872178), "SuryaBench SDO Dataset", 2025-12-09, licence `cc-by-4.0`. Its description says: "This repository exists only to register a DOI for the SuryaBench dataset hosted on the AWS Open Data Registry." It holds only `README.md` (280 bytes). Some author names are misspelled in the record itself: "gaur, Vishal", "Sila, Daniel da", "Freirag, Marcus", "Moreno, Juan Bernabe".

```bibtex
@dataset{roy_2025_17872179,
  author       = {Roy, Sujit and
                  Hegde, Dinesha Vasanta and
                  Schmude, Johannes and
                  Lal, Rohit and
                  gaur, Vishal and
                  Lin, Amy and
                  Mandal, Kshitiz and
                  Singh, Talwinder and
                  Munoz-Jaramillo, Andres and
                  Yang, Kang and
                  Pandey, Chetraj and
                  Hong, Jinsu and
                  Aydin, Berkay and
                  McGranaghan, Ryan and
                  Kasapis, Spiridon and
                  Upendran, Vishal and
                  Bahauddin, Shah and
                  Sila, Daniel da and
                  Freirag, Marcus and
                  Gurung, Iksha and
                  Pogorelov, Nikolai and
                  Watson, Campbell and
                  Maskey, Manil and
                  Moreno, Juan Bernabe and
                  Ramachandran, Rahul},
  title        = {SuryaBench SDO Dataset},
  month        = dec,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17872179},
  url          = {https://doi.org/10.5281/zenodo.17872179},
}
```
<sub>file: `bibtex/suryabench_zenodo_dataset_17872179.bib`</sub>

- **Presentation**: DOI 10.5281/zenodo.17407005, Hegde, "SuryaBench: Benchmark Dataset for Advancing Machine Learning Applications in Heliophysics and Space Weather", 2025-10-21, cc-by-4.0, file `DASH_2025_Talk_Hegde.pdf` (5079429 bytes, not downloaded).

```bibtex
@misc{hegde_2025_17407005,
  author       = {Hegde, Dinesha Vasanta},
  title        = {SuryaBench: Benchmark Dataset for Advancing
                   Machine Learning Applications in Heliophysics and
                   Space Weather
                  },
  month        = oct,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17407005},
  url          = {https://doi.org/10.5281/zenodo.17407005},
}
```
<sub>file: `bibtex/suryabench_zenodo_presentation_17407005.bib`</sub>

- Found by the same search but **not** a citation request: 10.5281/zenodo.22905499, "HELIOFLOOR: analysis code and data for an independent evaluation of Surya's solar flare forecasting" (Yıldırım, Kadir Can; 2026-09-22; v1.0.0; apache2.0; isSupplementTo 10.22541/essoar.15008581/v1). This may be useful context for the reproduction.

### 5.3 AWS Open Data Registry `surya-bench` (linked from the core-sdo card)
Source: `https://raw.githubusercontent.com/awslabs/open-data-registry/main/datasets/surya-bench.yaml`, lines 18–19 (the rendered page at https://registry.opendata.aws/surya-bench/ shows the same text under "How to Cite", preceded by "Surya Bench was accessed on DATE from https://registry.opendata.aws/surya-bench."):

```text
Citation: >
  Roy, S., Singh, T., Freitag, M., Schmude, J., Lal, R., Hegde, D., Ranjan, S., Lin, A., Gaur, V., Vos, E.E. and Ghosal, R., 2024. AI Foundation Model for Heliophysics: Applications, Design, and Implementation. arXiv preprint arXiv:2410.10841.
```

- This is a **different paper** (arXiv:2410.10841, v1 only, 30 Sep 2024, CC BY 4.0, "31 Pages, 12 figures"), not the SuryaBench paper. For convenience, its arXiv export is in `bibtex/aws_registry_cited_arxiv_2410.10841.bib`. The registry itself gives text only:

```bibtex
@misc{roy2024aifoundationmodelheliophysics,
      title={AI Foundation Model for Heliophysics: Applications, Design, and Implementation}, 
      author={Sujit Roy and Talwinder Singh and Marcus Freitag and Johannes Schmude and Rohit Lal and Dinesha Hegde and Soumya Ranjan and Amy Lin and Vishal Gaur and Etienne Eben Vos and Rinki Ghosal and Badri Narayana Patro and Berkay Aydin and Nikolai Pogorelov and Juan Bernabe Moreno and Manil Maskey and Rahul Ramachandran},
      year={2024},
      eprint={2410.10841},
      archivePrefix={arXiv},
      primaryClass={astro-ph.SR},
      url={https://arxiv.org/abs/2410.10841}, 
}
```
<sub>file: `bibtex/aws_registry_cited_arxiv_2410.10841.bib`</sub>

- Registry licence: "Creative Commons Attribution 4.0 International." Its `Documentation:` link `https://huggingface.co/datasets/nasa-impact/Surya-bench` returns **401** (renamed or removed). The registry text says coverage runs "to July 31, 2024", while the HF core-sdo card says "Dec 31, 2024".

---

## 6. TerraTorch — https://github.com/IBM/terratorch and https://ibm.github.io/terratorch/stable/
- **Location**: `github.com` HTML returns 403 through the proxy, but `git ls-remote https://github.com/IBM/terratorch.git` and `.../torchgeo/terratorch.git` both return HEAD `f967ef77b00fe9097157a29d1dad4fda93aba4f9`. The README links point to `torchgeo/terratorch`. Both facts fit a move of the repo to the `torchgeo` org. The blog's docs URL `https://ibm.github.io/terratorch/stable/` (and `/terratorch/`) returns **404**. `https://torchgeo.github.io/terratorch/stable/...` redirects (301) to `https://torchgeo.org/terratorch/stable/`.
- **CITATION.cff**: 404 at HEAD for both IBM and torchgeo paths (CITATION.bib, CITATION.md and `.zenodo.json` are also 404). **Zenodo**: the search `"TerraTorch"` returns 3 records, none of them TerraTorch itself.
- **README** (identical at HEAD and at tag `v1.2.13`). It has no BibTeX, only line 159:

```text
[TerraTorch: The Geospatial Foundation Models Toolkit on arXiv](https://arxiv.org/abs/2503.20563)
```

- **Docs "Citation" page** (`https://torchgeo.org/terratorch/stable/about/citation/`, source `docs/about/citation.md` on branch `docs` `b315bd90eec349a553139c47a642c4fe789686d1`; byte-identical at tag `v1.2.13` `d1267fbd278cd23bc05a1c22cc077e18481dd7c0` and `main` `f967ef77…`). Lines 3–5:

```text
We outlined the idea of TerraTorch and the overall architecture in our [paper](https://arxiv.org/abs/2503.20563).

If you are using TerraTorch, consider citing it with:
```

```bibtex
@article{gomes2025terratorch,
  title={TerraTorch: The Geospatial Foundation Models Toolkit},
  author={Gomes, Carlos and Blumenstiel, Benedikt and Almeida, Joao Lucas de Sousa and de Oliveira, Pedro Henrique and Fraccaro, Paolo and Escofet, Francesc Marti and Szwarcman, Daniela and Simumba, Naomi and Kienzler, Romeo and Zadrozny, Bianca},
  journal={arXiv preprint arXiv:2503.20563},
  year={2025}
}
```
<sub>file: `bibtex/terratorch_docs_citation.bib`</sub>

- Version: `https://torchgeo.org/terratorch/versions.json` lists `1.2.13` with alias `stable`. The latest tag from `git ls-remote` (sorted with `sort -V`) is `v1.2.13`. `pyproject.toml` has `version = "v1.2.13"`.
- arXiv 2503.20563 export (v1 only, 26 Mar 2025, CC BY-SA 4.0, comment "IGARSS 2025"):

```bibtex
@misc{gomes2025terratorchgeospatialfoundationmodels,
      title={TerraTorch: The Geospatial Foundation Models Toolkit}, 
      author={Carlos Gomes and Benedikt Blumenstiel and Joao Lucas de Sousa Almeida and Pedro Henrique de Oliveira and Paolo Fraccaro and Francesc Marti Escofet and Daniela Szwarcman and Naomi Simumba and Romeo Kienzler and Bianca Zadrozny},
      year={2025},
      eprint={2503.20563},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.20563}, 
}
```
<sub>file: `bibtex/terratorch_arxiv_2503.20563.bib`</sub>

- **Licence**: `pyproject.toml` `license = "Apache-2.0 AND MIT"`, `license-files = ["LICENSE", "MIT_FILES.txt"]`. README `## License` lines 191–193:

```text
This project is primarily licensed under the **Apache License 2.0**. 

However, some files contain code licensed under the **MIT License**. These files are explicitly listed in [`MIT_FILES.txt`](./MIT_FILES.txt).
```

---

## 7. The four arXiv papers linked in the blog
All exports come from `https://arxiv.org/bibtex/<id>`. The export always describes the **latest** version and has **no version field**. To pin a version, cite e.g. `arXiv:2508.14112v2`. Each paper also has a DataCite DOI `10.48550/arXiv.<id>` (shown on the abs page). Versions and licences come from the abs page "Submission history" and licence link, cross-checked with `export.arxiv.org/api/query` (`<id>` element ends in `vN`).

### 7.1 arXiv 2508.14112 — Surya: Foundation Model for Heliophysics
- v1 Mon, 18 Aug 2025 05:44:25 UTC (17,648 KB). **v2 Thu, 21 Aug 2025 16:53:36 UTC (17,517 KB) = latest**. Licence CC BY 4.0. Primary astro-ph.SR (also astro-ph.IM, cs.AI). DOI 10.48550/arXiv.2508.14112. No journal-ref.
- v1→v2 author change: v2 adds "Spalding, Anne" (33 authors vs 32).
- The paper's "Code and Data Availability" (v2 HTML) points to `https://huggingface.co/nasa-ibm-ai4science` and `https://github.com/NASA-IMPACT/Surya`. It gives no BibTeX.

```bibtex
@misc{roy2025suryafoundationmodelheliophysics,
      title={Surya: Foundation Model for Heliophysics}, 
      author={Sujit Roy and Johannes Schmude and Rohit Lal and Vishal Gaur and Marcus Freitag and Julian Kuehnert and Theodore van Kessel and Dinesha V. Hegde and Andrés Muñoz-Jaramillo and Johannes Jakubik and Etienne Vos and Kshitiz Mandal and Ata Akbari Asanjan and Joao Lucas de Sousa Almeida and Amy Lin and Talwinder Singh and Kang Yang and Chetraj Pandey and Jinsu Hong and Berkay Aydin and Thorsten Kurth and Ryan McGranaghan and Spiridon Kasapis and Vishal Upendran and Shah Bahauddin and Daniel da Silva and Nikolai V. Pogorelov and Anne Spalding and Campbell Watson and Manil Maskey and Madhulika Guhathakurta and Juan Bernabe-Moreno and Rahul Ramachandran},
      year={2025},
      eprint={2508.14112},
      archivePrefix={arXiv},
      primaryClass={astro-ph.SR},
      url={https://arxiv.org/abs/2508.14112}, 
}
```
<sub>file: `bibtex/surya_arxiv_2508.14112.bib`</sub>

### 7.2 arXiv 2508.14107 — SuryaBench
- **v1 only**: Mon, 18 Aug 2025 00:05:01 UTC (16,159 KB). CC BY 4.0. astro-ph.SR. DOI 10.48550/arXiv.2508.14107. Its "6 Code Availability" links the dead HF collection (§4) and `https://github.com/NASA-IMPACT/SuryaBench` (§5.1).

```bibtex
@misc{roy2025suryabenchbenchmarkdatasetadvancing,
      title={SuryaBench: Benchmark Dataset for Advancing Machine Learning in Heliophysics and Space Weather Prediction}, 
      author={Sujit Roy and Dinesha V. Hegde and Johannes Schmude and Amy Lin and Vishal Gaur and Rohit Lal and Kshitiz Mandal and Talwinder Singh and Andrés Muñoz-Jaramillo and Kang Yang and Chetraj Pandey and Jinsu Hong and Berkay Aydin and Ryan McGranaghan and Spiridon Kasapis and Vishal Upendran and Shah Bahauddin and Daniel da Silva and Marcus Freitag and Iksha Gurung and Nikolai Pogorelov and Campbell Watson and Manil Maskey and Juan Bernabe-Moreno and Rahul Ramachandran},
      year={2025},
      eprint={2508.14107},
      archivePrefix={arXiv},
      primaryClass={astro-ph.SR},
      url={https://arxiv.org/abs/2508.14107}, 
}
```
<sub>file: `bibtex/suryabench_arxiv_2508.14107.bib`</sub>

### 7.3 arXiv 2304.06446 — SpectFormer
- v1 Thu, 13 Apr 2023 12:27:17 UTC (1,128 KB). **v2 Fri, 14 Apr 2023 22:20:46 UTC (1,128 KB) = latest**. Licence **CC BY-NC-SA 4.0**. cs.CV. DOI 10.48550/arXiv.2304.06446. arXiv comment: "The project page is available at this webpage \url{https://badripatro.github.io/SpectFormers/}".

arXiv export:

```bibtex
@misc{patro2023spectformerfrequencyattentionneed,
      title={SpectFormer: Frequency and Attention is what you need in a Vision Transformer}, 
      author={Badri N. Patro and Vinay P. Namboodiri and Vijay Srinivas Agneeswaran},
      year={2023},
      eprint={2304.06446},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2304.06446}, 
}
```
<sub>file: `bibtex/spectformer_arxiv_2304.06446.bib`</sub>

Authors' own request, from the project page `https://badripatro.github.io/SpectFormers/` "Citation" (HTML `<code>` block). It is byte-identical to `badripatro/SpectFormers` README `## Citation` (HEAD `b6cf487a0ea71eeb252d5ba0fc7410b3f5dbd256`, lines 46–51). Note the double space after "Badri N.":

```bibtex
@article{patro2023spectformer,
  title={SpectFormer: Frequency and Attention is what you need in a Vision Transformer},
  author={Patro, Badri N.  and Namboodiri, Vinay P. and Agneeswaran, Vijay Srinivas},
  journal={arXiv preprint arXiv:2304.06446},
  year = {2023}
}
```
<sub>file: `bibtex/spectformer_project_page.bib`</sub>

Code licence: no LICENSE file at HEAD (`LICENSE`, `LICENSE.md`, `LICENSE.txt` and `license` all 404).

### 7.4 arXiv 2107.02192 — Long-Short Transformer (blog links `https://arxiv.org/pdf/2107.02192`)
- v1 Mon, 5 Jul 2021 18:00:14 UTC (486 KB). v2 Tue, 27 Jul 2021 19:34:30 UTC (488 KB). **v3 Tue, 7 Dec 2021 07:23:24 UTC (503 KB) = latest**. Licence **CC BY-NC-SA 4.0**. cs.CV. DOI 10.48550/arXiv.2107.02192. Comment: "Published at NeurIPS 2021".

```bibtex
@misc{zhu2021longshorttransformerefficienttransformers,
      title={Long-Short Transformer: Efficient Transformers for Language and Vision}, 
      author={Chen Zhu and Wei Ping and Chaowei Xiao and Mohammad Shoeybi and Tom Goldstein and Anima Anandkumar and Bryan Catanzaro},
      year={2021},
      eprint={2107.02192},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2107.02192}, 
}
```
<sub>file: `bibtex/long-short-transformer_arxiv_2107.02192.bib`</sub>

- Code `NVIDIA/transformer-ls` (linked from the abs page; HEAD `1527c5cc760daf1432a5c5c1fdc2ffb2f074c969`): the README only links the arXiv paper and has **no BibTeX or citation section**. LICENSE: MIT, "Copyright (c) 2021 NVIDIA CORPORATION.".
- I did **not** get a NeurIPS 2021 proceedings BibTeX: DBLP (`dblp.org/search/publ/api`) returned a bot-check page (HTTP 202), and I did not guess the proceedings URL.

---

## 8. Discrepancies to keep in mind when citing
1. **Surya has three BibTeX variants** (README `@article`, 10 authors + others; HF `@misc` with a placeholder eprint, 32 authors; arXiv `@misc`, 33 authors). For a correct reference, the arXiv export (v2) plus the git commit/HF sha you actually used is the most complete option. The README's `@article ... journal={arXiv preprint ...}` is the form the code authors ask for.
2. The **HF Surya-1.0 card BibTeX is broken as published**: `eprint={XXXX.XXXXX}`, and its code fence is not closed.
3. **SuryaBench author lists differ**: arXiv 25, GitHub README 21, Zenodo dataset record 25 but with misspellings.
4. The **AWS registry asks for a different paper** (arXiv:2410.10841, 2024) than the SuryaBench paper.
5. **Version labels do not line up**: Surya git tag `v1.0.0` (the README there has no citation), pyproject `0.1.0`, HF weights `surya.366m.v1`, and the HF repo has no tags. HF DOIs exist only for the SuryaBench datasets, not for the Surya model.
6. **Dead links in the blog and docs**: SuryaBench collection (401), `ibm.github.io/terratorch/stable/` (404), AWS registry documentation link (401), SuryaBench README collection link (401).
7. The Lunar/SomBench dataset-card `fraccaro2026lfm` entry has a BibTeX syntax error (missing comma).

## 9. What could not be reached or verified
- **GitHub REST API** for `NASA-IMPACT/Surya` (`/releases`, `/tags`, `/repos/...`, `/contents/`) and the github.com HTML pages: HTTP 403 from the session proxy ("GitHub access to this repository is not enabled for this session"), or 400 for `/contents/`. Release objects (names, notes, assets, dates) are **unknown**. I used `git ls-remote` for tags instead. I did not use credentials or tokens.
- **HF collection** `nasa-impact/suryabench-68265ce306fc2470c121af7b` (the blog's link): 401, no archive copy.
- **`https://ibm.github.io/terratorch/stable/`**: 404 (I used the torchgeo.org docs instead).
- **DBLP**: bot-check page, so no NeurIPS BibTeX for the Long-Short Transformer.
- Zenodo checks are **keyword searches**. "No record" means none was found by those queries.
- HF's own "Cite this dataset" UI is rendered by JavaScript and was not fetched. The DOI BibTeX above comes from DOI content negotiation (crosscite), which may be formatted differently from HF's UI.
- No model weights, datasets or PDFs were downloaded. All files fetched were small text/JSON/HTML; the largest was the 344,118-byte arXiv HTML of 2508.14112v2.

## 10. Files
- Notes: `/home/user/work/surya_scout/research/citations.md` (this file)
- Verbatim BibTeX: `/home/user/work/surya_scout/research/bibtex/*.bib` (index with sources: `/home/user/work/surya_scout/research/bibtex/SOURCES.tsv`)
- Raw sources: `/home/user/work/surya_scout/research/sources/{github,hf,doi,arxiv,docs,zenodo,aws,dblp}/`
