# Emulation
This contains the kriging code. It is written as an object, 
so it can easily be added into a program of your own. An example of 
how to use it is in the unit test.

At the moment, Emulator.py does kriging with a single kriging length.
Emulator_v3.py is a version in development that will have a different
kriging length for each element of the input array (x_star).