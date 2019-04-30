# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:27:16 2019

Download these libraries your Anaconda Virtual Environment through the GUI
- math
- numpy
- tensorflow

@author: Greg
"""

#------------------------------------math--------------------------------------
print('------------------------------------math--------------------------------------')

# The import function allows you to load libraries
import math
print("It's math! It has type {}".format(type(math)))

# What functions are in this library?
print('--------------------------------------------------')
print(dir(math))

# Example of how to use pi
print('--------------------------------------------------')
print("pi to 4 significant digits = {:.4}".format(math.pi))


# Example of how to log base something
print('--------------------------------------------------')
print(math.log(32, 2))

# Can also use the help() function once loaded
print('--------------------------------------------------')
help(math.log)

# We can also import one thing at a time from math and use any name we want
print('--------------------------------------------------')
import math as mt
print(mt.pi)

# Put it back now for clarity
import math
mt = math

# We can use this syntax to not have to do the module.variable thing
print('--------------------------------------------------')
from math import *
print(pi, log(32, 2))
# but this isn't really good practice - it's better to use just what you need
# Also, note the little warning triangles - spyder is confused that we keep importing math
# But it's not so confused that it won't still run for us

#------------------------------------numpy-------------------------------------
print('------------------------------------numpy-------------------------------------')

import numpy
print("numpy.random is a", type(numpy.random))
print("it contains names such as...",
      dir(numpy.random)[-15:]
      )

# Submodule call - roll 10 dice
print('--------------------------------------------------')
rolls = numpy.random.randint(low=1, high=6, size=10)
print(rolls)
print()
print(type(rolls))
print()
print(dir(rolls))

# Can convert a numpy array to a list
print('--------------------------------------------------')
rolls = rolls.tolist()
print(rolls)
print()
print(type(rolls))


# Put it back to a numpy array
rolls = numpy.array(rolls)
print(rolls)
print()
print(type(rolls))

# Help on stuff
print('--------------------------------------------------')
help(rolls.ravel)

# Can do vector and matrix manipulations with numpy arrays
print('--------------------------------------------------')
print(rolls + 10)

# At which indices are the dice less than or equal to 3?
print(rolls <= 3)

# Create a matrix
print('--------------------------------------------------')
xlist = [[1,2,3],[2,4,6],]
# Create a 2-dimensional array
x = numpy.asarray(xlist)
print("xlist = {}\nx =\n{}".format(xlist, x))
      
#----------------------------------TensorFlow----------------------------------
print('---------------------------------TensorFlow----------------------------------')

# When does 1+1 not equal 2?
import tensorflow as tf
# Create two constants, each with value 1
a = tf.constant(1)
b = tf.constant(1)
# Add them together to get...
print(a + b)