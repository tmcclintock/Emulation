Emulation
=========
This is a short and simple imlpementation of basic Gaussian-Process
(aka kriging) code. In the cosmology community it is known as
an emulator, hence the name.

Installation
------------
To install simply write
```
python setup.py install
```
if you care about keeping the root directory clean then do
```
python setup.py clean
```

Usage
-----
An example of how to use this code can be found
in the **examples/example.py** file. The result is
an emulator which interpolates over the region
where the data was generated, which can be seen
in the following

![alt text](https://github.com/tmcclintock/Emulation/blob/master/figures/emulator_example.png)