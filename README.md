# Sessile Drop Analyzer

[![DOI](https://zenodo.org/badge/1375806084.svg)](https://doi.org/10.5281/zenodo.22830731)

A Python-based tool for analysing sessile drop profiles from experimental images using the Young–Laplace equation.

The program determines the main geometrical characteristics of the sessile drop, including its **height, maximum diameter, and contact diameter**, and estimates the **surface tension** of the liquid.

## How It Works

The analysis is performed in several steps:

1. **Import the image**
   The user imports an experimental image containing the sessile drop on a substrate.

2. **Enter experimental parameters**
   The user specifies:

   * the volume of the sessile drop;
   * the density difference between the drop and the surrounding medium.

3. **Select the threshold**
   A threshold value is selected to distinguish the **drop and substrate from the background**. This allows the programme to identify the relevant contours in the image.

4. **Define the drop contour**
   After closing the threshold selection window, the programme detects the contours in the image. The user then identifies the contour corresponding to the sessile drop and separates it from the substrate.

5. **Select reference points**
   The user defines three points on the drop:

   * the **starting point** of the sessile drop, e.g. the left contact point between the drop and substrate;
   * the **end point** of the sessile drop, e.g. the right contact point between the drop and substrate;
   * an additional point located on the drop profile.

6. **Calculate drop properties**
   After closing the selection window, the programme performs the calculation. It determines the **drop height, maximum diameter, and contact diameter**, and estimates the **surface tension** using the Young–Laplace equation.

## Requirements

* Python 3.11
* NumPy
* SciPy
* Matplotlib
* OpenCV (`cv2`)

## Installation

Install the required Python packages using `pip`:

```bash
pip install numpy scipy matplotlib opencv-python
```
