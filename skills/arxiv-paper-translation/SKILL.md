# arXiv Paper Translation Workflow

Use this when the user asks to learn an arXiv paper by downloading its source into `papers/` and creating a Markdown preview.

## Goal

Create a Chinese Markdown translation that follows the original paper order, with figures placed near their original positions. This is not a high-level summary or reading guide.

## Workflow

1. Download the arXiv source package from `https://arxiv.org/e-print/<id>` and extract it under:

   `papers/<paper title>/`

2. Read the source entry file from `00README.json`. Usually the top-level file is `main.tex`, `paper.tex`, or `example_paper.tex`.

3. Translate only the active paper text. Ignore commented-out LaTeX drafts, hidden notes, review notes, and unused sections.

4. Create or update:

   `papers/<paper title>/summary.md`

5. Keep the translation in the original order:

   - front matter
   - abstract
   - introduction
   - method sections
   - evaluation
   - related work
   - conclusion
   - useful appendix material

6. Convert figure PDFs into PNGs for editor preview. Put them in:

   `papers/<paper title>/figures_png/`

   On macOS, `qlmanage -t -s 2200` gives clear previews. `sips` works as a fallback but may produce small images.

7. Insert figures near their original LaTeX positions. Use Markdown image attributes to keep preview size reasonable:

   ```markdown
   ![](figures_png/fig1_scheduling_models.png){width=78%}
   ```

   Suggested widths:

   - single-column figures: `78%` to `82%`
   - wide overview figures: `88%` to `90%`
   - evaluation plots: around `80%`

## Markdown Style

Use this front matter shape:

```yaml
---
title: Paper Title - 原文翻译
mathjax: true
categories:
  - 编译器
date: YYYY-MM-DD HH:mm:ss
tags:
  - Paper
---
```

Preserve important equations in MathJax. Translate technical terms consistently, keeping common English terms when they are clearer, such as `kernel`, `megakernel`, `Event Tensor`, `shape dynamism`, `data-dependent dynamism`, `static scheduling`, and `dynamic scheduling`.

## Checks Before Finishing

- Every image referenced in `summary.md` exists.
- Figures are readable but not oversized in Markdown preview.
- The result reads like a paper translation, not a summary.
- The original source files remain untouched except for adding `summary.md` and generated preview images.
