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

I am sorry it has taken me so long to get back to you. The past few weeks have been very busy for me. I have now finished working through the Surya blog post, and I have attached my report as a PDF.

The blog post itself has no Citation button, so I opened every tool and paper it links to and recorded the citation each one asks for. The tools are the Surya code, the Surya-1.0 model and four fine-tuned models on Hugging Face, the SuryaBench datasets, and IBM's TerraTorch library. I downloaded all of them.

Pretraining and fine-tuning need many GPUs, so I could not repeat any training on my CPU machine. Instead I ran the released weights on everything that only needs them. The repository's validation test passes on CPU (mean loss 0.316689 against the reference 0.316647). Zero-shot forecasting on 20 times from the paper's validation period gives a 1-hour MSE of 0.195, close to the 0.202 in Table 9. For the four downstream tasks I could only afford small samples, 40 to 70 cases each. On these the paper's numbers for active-region segmentation and flare forecasting fall inside my 95% intervals. The solar wind model agrees with the paper on the validation period but not on 2020 to 2024, where it underestimates the speed and does worse than simply predicting the average speed of the training years. For the EUV spectra the paper's numbers fall between my two samples. The paper does not say which split it used, and the flare numbers suggest it was the validation period.

A few things surprised me. Surya gives wrong results on CPU in bfloat16, which I traced to a PyTorch convolution bug that hits exactly its first layer. When it is loaded through TerraTorch with default settings, the base model runs with a different number of attention heads than the released config, and the fine-tuned models do not load the weights learned for their tasks, in both cases without any warning. And the first layer of the released EUV model was never trained; it is still the random initialisation. The report lists a few other places where the released code seems to differ from the paper. I may well be missing something there, so I would be grateful for any comments.

Thank you again for your guidance.

Best regards,
Xiangsheng (Harry) Bao
