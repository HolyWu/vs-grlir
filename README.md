# GRL for Image Restoration
Efficient and Explicit Modelling of Image Hierarchies for Image Restoration, based on https://github.com/ofsoundof/GRL-Image-Restoration.


## Dependencies
- [NumPy](https://numpy.org/install)
- [PyTorch](https://pytorch.org/get-started) 1.13.1
- [VapourSynth](http://www.vapoursynth.com/) R55+


## Installation
```
pip install -U vsgrlir
python -m vsgrlir
```


## Usage
```python
from vsgrlir import grlir

ret = grlir(clip)
```

See `__init__.py` for the description of the parameters.
