# Scribble-Supervised Learning for Medical Image Segmentation

This repository provides re-implementations of some papers about scribble-supervised  for medical image segmentation:

## Related Papers

1. [DMPLS](https://arxiv.org/pdf/2203.02106) — *MICCAI 2022*  ✅ 

2. [ScribbleVC](https://arxiv.org/pdf/2307.16226) — *ACM MM 2023*  ✅

3. [ScribFormer](https://arxiv.org/pdf/2402.02029) — *IEEE TMI 2024*  ✅

4. [DMSPS](https://www.sciencedirect.com/science/article/abs/pii/S1361841524001993?dgcid=author) — *MedIA 2024*  ✅ 

5. [ScribbleVS](https://arxiv.org/pdf/2411.10237) — *arXiv 2024*  ✅ 

6. [AIL](https://ieeexplore.ieee.org/abstract/document/10851813) - *IEEE TIP 2025* Fix bug, not run

7. [TABNet](https://arxiv.org/pdf/2507.02399) — *arXiv 2025* ✅


## 📊  Benchmark on ACDC

<table style="border-collapse: collapse; text-align: center;">
  <thead>
    <tr>
      <th style="border: 1px solid black; padding: 6px;">Method</th>
      <th style="border: 1px solid black; padding: 6px;">LV<br>(Dice / HD95)</th>
      <th style="border: 1px solid black; padding: 6px;">RV<br>(Dice / HD95)</th>
      <th style="border: 1px solid black; padding: 6px;">MYO<br>(Dice / HD95)</th>
      <th style="border: 1px solid black; padding: 6px;">Mean<br>(Dice / HD95)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid black; padding: 6px;"><b>TABNet</b></td>
      <td style="border: 1px solid black; padding: 6px;">87.55 / 1.49</td>
      <td style="border: 1px solid black; padding: 6px;">88.87 / 5.10</td>
      <td style="border: 1px solid black; padding: 6px;">92.65 / 6.47</td>
      <td style="border: 1px solid black; padding: 6px;"><b>89.69 / 4.35</b></td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 6px;"><b>ScribbleVS</b></td>
      <td style="border: 1px solid black; padding: 6px;">88.79 / 1.32</td>
      <td style="border: 1px solid black; padding: 6px;">88.70 / 1.14</td>
      <td style="border: 1px solid black; padding: 6px;">93.13 / 1.16</td>
      <td style="border: 1px solid black; padding: 6px;"><b>90.21 / 1.21</b></td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 6px;"><b>DMSPS</b></td>
      <td style="border: 1px solid black; padding: 6px;">87.83 / 2.09</td>
      <td style="border: 1px solid black; padding: 6px;">87.39 / 1.21</td>
      <td style="border: 1px solid black; padding: 6px;">92.54 / 1.23</td>
      <td style="border: 1px solid black; padding: 6px;">89.25 / 1.51</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 6px;"><b>DMPLS</b></td>
      <td style="border: 1px solid black; padding: 6px;">87.27 / 1.38</td>
      <td style="border: 1px solid black; padding: 6px;">87.13 / 1.31</td>
      <td style="border: 1px solid black; padding: 6px;">92.20 / 4.87</td>
      <td style="border: 1px solid black; padding: 6px;">88.87 / 2.52</td>
    </tr>
    <tr>
      <td style="border: 1px solid black; padding: 6px;"><b>ScribbleVC</b></td>
      <td style="border: 1px solid black; padding: 6px;">84.92 / 1.55</td>
      <td style="border: 1px solid black; padding: 6px;">85.50 / 1.46</td>
      <td style="border: 1px solid black; padding: 6px;">91.05 / 2.05</td>
      <td style="border: 1px solid black; padding: 6px;">87.16 / 1.68</td>
    </tr>

  </tbody>
</table>

- chạy thêm ACDC, MSCMRseg
- đọc kĩ ScribbleVS
- slide mô tả chi tiết ScribbleVS
- thiết kế thực nghiệm ablation


<!-- 

### Tasks
- Test ScribbleVC, Scribformer, DMSPS stage2
- Run AIL

### Acknowledgement
This repo partially uses code from [Hilab-WSL4MIS](https://github.com/HiLab-git/WSL4MIS) and [ShapePU](https://github.com/BWGZK/ShapePU) -->