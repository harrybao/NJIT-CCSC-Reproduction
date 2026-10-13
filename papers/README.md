# Papers

* PDFs are **not** redistributed here.
* Full texts were read from arXiv (14 papers used for comparison; the 2 MagNet papers were downloaded but not analysed), *Scientific Reports* (SolarFlareNet, FlareDB) and *Frontiers* (CMETNet).
* Not accessible from the environment: A&A (DeepHMI), the Springer LNCS chapter (KpNet), and AGU/Wiley (SINet's JGR version; its arXiv version was read).
* What each tool's **Citation button** points to (a Zenodo software archive, not the paper) is in [`citations/`](citations/README.md).

| Tool | Reference | DOI | Open-access copy | Read here |
|---|---|---|---|---|
| LSTM-flare-prediction | Liu H., Liu C., Wang J.T.L., Wang H. 2019, ApJ 877:121 | 10.3847/1538-4357/ab1b3c | arXiv:1905.07095 | ✅ |
| RNN-CME-prediction | Liu H., Liu C., Wang J.T.L., Wang H. 2020, ApJ 890:12 | 10.3847/1538-4357/ab6850 | arXiv:2002.10953 | ✅ |
| FlareML / DeepSun | Abduallah Y. et al. 2021, RAA 21(7):160 | 10.1088/1674-4527/21/7/160 | arXiv:2009.04238 | ✅ |
| SolarFlareNet | Abduallah Y., Wang J.T.L., Wang H., Xu Y. 2023, Sci. Rep. 13:13665 | 10.1038/s41598-023-40884-1 | open access (nature.com) | ✅ |
| SEP-prediction | Abduallah Y. et al. 2022, ApJS 260:16 | 10.3847/1538-4365/ac5f56 | arXiv:2203.14393 | ✅ |
| CMETNet | Alobaid K.A. et al. 2022, Front. Astron. Space Sci. 9:1013345 | 10.3389/fspas.2022.1013345 | open access | ✅ |
| Dst (DSTT) | Abduallah Y. et al. 2022, FLAIRS-35 | 10.32473/flairs.v35i.130564 | arXiv:2205.02447 | ✅ (results in figures) |
| Kp (KpNet) | Abduallah Y., Wang J.T.L., Xu C., Wang H. 2022, ISMIS 2022 (LNCS 13515, 325–335) | 10.1007/978-3-031-16564-1_31 | — | ✗ |
| TSInet | Abduallah Y. et al. 2021, FLAIRS-34 | 10.32473/flairs.v34i1.128356 | arXiv:2107.11042 | ✅ (results in figures) |
| SYMHnet | Abduallah Y. et al. 2024, Space Weather 22, e2023SW003824 | 10.1029/2023SW003824 | arXiv:2402.17196 | ✅ |
| SolarUnet | Jiang H. et al. 2020, ApJS 250:5 | 10.3847/1538-4365/aba4aa | arXiv:2008.12080 | ✅ |
| FibrilNet | Jiang H. et al. 2021, ApJS 256:20 | 10.3847/1538-4365/ac14b7 | arXiv:2107.07886 | ✅ |
| SDNN | Jiang H. et al. 2022, ApJ 939:66 | 10.3847/1538-4357/ac927e | arXiv:2210.04122 | ✅ |
| SolarCNN | Xu C. et al. 2024, Sol. Phys. 299:36 | 10.1007/s11207-024-02283-1 | arXiv:2403.18302 | ✅ |
| DeepHMI | Xu C., Xu Y., Wang J.T.L., Li Q., Wang H. 2025, A&A | 10.1051/0004-6361/202453581 | — | ✗ |
| SINet | Wang Z. et al. 2026, JGR Space Physics | 10.1029/2025JA034868 | arXiv:2604.10045 | ✅ (arXiv) |
| DeepGeoCME | Yan Z., Wang J.T.L., Wang H., Lee H., Jing J., Xu Y., Xu C., Yurchyshyn V. 2026 | — | arXiv:2605.24748 | ✅ |
| MIST (closest paper; link inferred) | Chaudhary Y., Wang J.T.L., Xu C., Xu Y., Zhang S. 2026, *Prediction of Solar Flares Using Photospheric Magnetic Field Parameters with Deep Learning* | — | arXiv:2606.21896 | ✅ |
| LLMFlare | no paper found (software record only: Zenodo 10.5281/zenodo.20738748) | — | — | — |
| MagNet | Jiang H. et al. 2023, Sol. Phys. 298:87 | 10.1007/s11207-023-02180-z | arXiv:2211.02278 | downloaded; not analysed (MagNet not run) |
| MagNet validation | Yesmanchyk A. et al. 2026, "Out-of-Sample Validation of MagNet" | — | arXiv:2601.15926 | downloaded; not analysed |
| FlareDB (CCDC) | Liu N. et al. 2026, Scientific Data | 10.1038/s41597-026-06607-7 | open access | ✅ |
| SolarDB white paper | Nita G. et al. 2022 | — | arXiv:2203.09544 | — |
| AI-ready data | Poduval B. et al. 2023, Front. Astron. Space Sci. 10 | 10.3389/fspas.2023.1203598 | open access | — |

To fetch the arXiv PDFs locally: `curl -L -o papers/<id>.pdf https://arxiv.org/pdf/<id>`.
