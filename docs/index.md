---
title: Home
---

# **MoPAnDA** 

**<big>Mo</big>**dular **<big>P</big>**etrophysical **<big>A</big>**nalysis a**<big>n</big>**d **<big>D</big>**ata **<big>A</big>**nalysis Tool

---
## What is MoPAnDA?

**MoPAnDA** is a petrophysics with python software built with **graphyical user interface** (**GUI**), allowing scientific python computing of conventional and unconventional formation evaluation. 
Much of its modules are designed to be a complimentary scoping tool for CCUS/CCS projects.

**MoPAnDA** can achieve following functions through GUI:

- Load and output data from `.las`, `.dlis`, `.xlsx`, `.csv` files.
- Editing and displaying logs in realtime.
- Data imputation and explorative analysis.
- Superwised and unsupervised prediction of missing logs and electrofacies.
- Petrophysical workflow. 

---
## On the Shoulders of Giants 

This project is based on multiple opensource packages:

* [lasio]: .LAS file reading and writing. lasio_ also provides standard output data structure across many petrophysical data processing softwares ([welly], [PetroPy]...)
* [PetroPy]: Funcdation of this project but sadly it's no longer updated and supported. MoPanda harvests the reservoir fluid property and multimineral model functions from [PetroPy].
* [PfEFFER] by KGS: [PfEFFER] is part of the KGS [GEMINI] project (currently finished development with no technical support) which created a complex petrophysical and geological processing software based on Java. However, it was talored to the complete and well-maintained database KGS has, thus lacking flexibility.








[lasio]: https://github.com/kinverarity1/lasio
[PetroPy]: https://github.com/toddheitmann/petropy
[welly]: https://github.com/agilescientific/welly
[PfEFFER]: https://www.kgs.ku.edu/software/PfEFFER-java/index.html
[GEMINI]: https://www.kgs.ku.edu/Gemini/Tools/Tools.html


<style>
big { color: #338eee; font-size:30px }
small { color: yellow }
</style>