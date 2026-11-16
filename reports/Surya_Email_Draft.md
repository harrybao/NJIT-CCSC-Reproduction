# Draft reply to Prof. Wang about Surya (for Harry to review, NOT sent)

> 中文说明（给 Harry）：
> * 下面是一封简短的回复，建议回复在教授那封邮件的同一个线程里，并附上报告 PDF（`reports/pdf/Surya_Reproduction_Report.pdf`）或仓库链接。
> * 邮件只放最重要的几点：博客没有 Citation 按钮、能跑的和不能跑的、主要结果、两个意外发现。细节全在报告里。
> * 措辞原则和上次一样：先说做了什么，问题写成"我观察到"，不下"论文有错"的结论。
> * 正文按"附 PDF"写的：发送时把 `reports/pdf/Surya_Reproduction_Report.pdf` 作为附件。如果你想改成给仓库链接，把第一段最后一句换成链接即可；仓库是私有的话，教授打不开，所以建议附 PDF。
> * 发送前请自己通读一遍，按你的习惯改语气。

---

Subject: Re: (same thread as Prof. Wang's email)

Dear Professor Wang,

I am sorry it has taken me so long to get back to you; the past few weeks have been very busy. I have now finished the Surya work, and my report is attached.

The blog has no Citation button, so I recorded the citation that each linked tool asks for. I could not repeat any training without GPUs, but I ran all the released models on CPU. The repository's test passes, and the zero-shot forecasts match Table 9. On small samples, the downstream results agree with the paper for active-region segmentation and flare forecasting. The solar wind model does not hold up on 2020 to 2024 data. I also found a few problems in the released code and in the TerraTorch version, which are listed in the report.

Thank you again for your guidance.

Best regards,
Xiangsheng (Harry) Bao
