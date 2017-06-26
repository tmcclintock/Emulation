Emulation
=========
This is a pure python imlpementation of basic Gaussian-Process
(aka kriging) code. In the cosmology community it is known as
an emulator, hence the name. This only implements the 
squared-exponential kernel, so it is less flexible than
george. On the other hand it's dependencies are much less
restrictive than george which can only be easily run on 
a Mac or Linux machine.

Installation
------------
To install write
```
python setup.py install
```
if you care about keeping the root directory clean then do
```
python setup.py clean
```

Usage
-----
Examples of how to use this code can be found
in the **examples/** directory. From the **cosine_example.py**
example you should get

![alt text](https://github.com/tmcclintock/Emulation/blob/master/figures/cosine_example.png)

while for the **reaction_rate_example.py** you should get

![alt text](https://github.com/tmcclintock/Emulation/blob/master/figures/reaction_rate.png)
