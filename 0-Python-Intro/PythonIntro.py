# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:39:29 2019

You don't need any libraries or anything to run this in Spyder.
Just download Anaconda, install spyder, and click "Run file (F5)"

Compare input to output and read the comments (the #'d lines) to intro Python!

@author: Greg Loughnane
"""

#-----------------------------Hello (Monty) Python-----------------------------    
print('-----------------------------Hello (Monty) Python----------------------------- ')

# Variable assignment
spam_amount = 0
print(spam_amount)

# Ordering Spam, egg, Spam, Spam, bacon and Spam (4 more servings of Spam)
print('--------------------------------------------------')
spam_amount = spam_amount + 4
print(spam_amount)

# Conditional logic (easy) - see Conditionals below
print('--------------------------------------------------')
if spam_amount > 0:
    print("But I don't want ANY spam!")

# Use the conditional logic statement
print('--------------------------------------------------')
viking_song = "Spam " * spam_amount
print(viking_song)

# Check variable types
print('--------------------------------------------------')
spam_amount = 0
print(type(spam_amount))
print(type(19.95))

# Working with operators
# Divide
print('--------------------------------------------------')
print(5 / 2)
print(6 / 2)

# Divide and round down
print('--------------------------------------------------')
print(5 // 2)
print(6 // 2)

# PEMDAS
print('--------------------------------------------------')
print(8 - 3 + 2)
print(-3 + 4 * 2)

# How tall am I, in meters, when wearing my hat?
print('--------------------------------------------------')
hat_height_cm = 25
my_height_cm = 190
total_height_meters = hat_height_cm + my_height_cm / 100
print("Height in meters =", total_height_meters, "?")
# Better...
total_height_meters = (hat_height_cm + my_height_cm) / 100
print("Height in meters =", total_height_meters)

# Minimum and maximum
print('--------------------------------------------------')
print(min(1, 2, 3))
print(max(1, 2, 3))

# Absolute value
print('--------------------------------------------------')
print(abs(32))
print(abs(-32))

# int and float are functions too
print('--------------------------------------------------')
print(float(10))
print(int(3.33))

# They can even be called on strings!
print('--------------------------------------------------')
print(int('807') + 1)

#-----------------------------Functions and help()-----------------------------
print('#-----------------------------Functions and help()-----------------------------')    

# Since help produces an obnoxiously long output, uncomment the next line by 
# using the hot key Ctrl+1 and then run this code
help(print)
help(round)

# Define a function
print('--------------------------------------------------')
def least_difference(a, b, c):
    """Return the smallest difference between any two numbers
    among a, b and c.
    
    e.g., least_difference(1, 5, -5)
    
    will give the output...
    
    4
    """
    diff1 = abs(a - b)
    diff2 = abs(b - c)
    diff3 = abs(a - c)
    return min(diff1, diff2, diff3)

print(
    least_difference(1, 10, 100),
    least_difference(1, 10, 10),
    least_difference(5, 6, 7) ) # Use trailing commas to go to the next line

# If we write in the green text above in our function, we can see it via help()
help(least_difference)

# There are additional arguments that we can use within functions
print('--------------------------------------------------')
print(1, 2, 3, sep=' < ')

# And we can make functions that have these additional arguments
print('--------------------------------------------------')
def greet(who="Colin"):
    print("Hello,", who)
    
greet()
greet(who="Kaggle")
greet("world")

# Meta-functions/functions within functions
print('--------------------------------------------------')
def mult_by_five(x):
    return 5 * x

def call(fn, arg):
    """Call fn on arg"""
    return fn(arg)

def squared_call(fn, arg):
    """Call fn on the result of calling fn on arg"""
    return fn(fn(arg))

print(
    call(mult_by_five, 1),
    squared_call(mult_by_five, 1), 
    sep='\n', # '\n' is the newline character - it starts a new line
)

print('--------------------------------------------------')
def mod_5(x):
    """Return the remainder of x after dividing by 5"""
    return x % 5

print(
    'Which number is biggest?',
    max(100, 51, 14),
    'Which number is the biggest modulo 5?',
    max(100, 51, 14, key=mod_5),
    sep='\n',
)

#----------------------------------Booleans-----------------------------------
print('----------------------------------Booleans-----------------------------------')

# This is one way to make a boolean
x = True
print(x)
print(type(x))

# Booleans can be used to test things
print('--------------------------------------------------')
print(3.0 == 3)

# Create a function to decide if I've got an odd number
print('--------------------------------------------------')
def is_odd(n):
    return (n % 2) == 1

print("Is 100 odd?", is_odd(100))
print("Is -1 odd?", is_odd(-1))

# Create a function to determine if someone can run for president
print('--------------------------------------------------')
def can_run_for_president(age):
    """Can someone of the given age run for president in the US?"""
    # The US Constitution says you must "have attained to the Age of thirty-five Years"
    return age >= 35

print("Can a 19-year-old run for president?", can_run_for_president(19))
print("Can a 45-year-old run for president?", can_run_for_president(45))

# Maybe we should include nationality
print('--------------------------------------------------')
def can_run_for_president(age, is_natural_born_citizen):
    """Can someone of the given age and citizenship status run for president in the US?"""
    # The US Constitution says you must be a natural born citizen *and* at least 35 years old
    return is_natural_born_citizen and (age >= 35)

print(can_run_for_president(19, True))
print(can_run_for_president(55, False))
print(can_run_for_president(55, True))

# Try this one
print('--------------------------------------------------')
print(True or True and False)

# Turn stuff into booleans
print('--------------------------------------------------')
print(bool(1)) # all numbers are treated as true, except 0
print(bool(0))
print(bool("asf")) # all strings are treated as true, except the empty string ""
print(bool(""))
# Generally empty sequences (strings, lists, and other types we've yet to see like lists and tuples)
# are "falsey" and the rest are "truthy"

#--------------------------------Conditionals----------------------------------
print('--------------------------------Conditionals----------------------------------')

# This is conditional logic with full coverage and beyond
def inspect(x):
    if x == 0:
        print(x, "is zero")
    elif x > 0:
        print(x, "is positive")
    elif x < 0:
        print(x, "is negative")
    else:
        print(x, "is unlike anything I've ever seen...")

inspect(0)
inspect(-15)

# This is conditional logic with full coverage and beyond
print('--------------------------------------------------')
def f(x):
    if x > 0:
        print("Only printed when x is positive; x =", x)
        print("Also only printed when x is positive; x =", x)
    print("Always printed, regardless of x's value; x =", x)

f(1)
f(0)

# Python is smart enough to use booleans without being told
print('--------------------------------------------------')
if 0:
    print(0)
elif "spam":
    print("spam")
    
# Use this if you have some grading to do
print('--------------------------------------------------')
def quiz_message(grade):
    if grade < 50:
        outcome = 'failed'
    else:
        outcome = 'passed'
    print('You', outcome, 'the quiz with a grade of', grade)
    
quiz_message(80)

# Or if you like combining logic into a single line...
def quiz_message(grade):
    outcome = 'failed' if grade < 50 else 'passed'
    print('You', outcome, 'the quiz with a grade of', grade)
    
quiz_message(45)

#-----------------------------------Lists--------------------------------------
print('-----------------------------------Lists--------------------------------------')

# Create lists
primes = [2, 3, 5, 7]
print(primes)

planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
print(planets)

hands = [
    ['J', 'Q', 'K'],
    ['2', '2', '2'],
    ['6', 'A', 'K'], # (Comma after the last element is optional)
]
print(hands)

my_favourite_things = [32, 'raindrops on roses', help]
print(my_favourite_things)

# Get stuff out of the planet list
print('--------------------------------------------------')
print(planets[0])   # Closest to sun
print(planets[1])   # Second closest to sun
print(planets[-1])  # Furthest from the sun
print(planets[-2])  # Second furthest from the sun
print(planets[0:3]) # The three closest to the sun
print(planets[:3])  # Same
print(planets[3:])  # All but the three closest to the sun
print(planets[1:-1])# All but the closest and the furthest
print(planets[-3:]) # The last three planets

# Rename Mars
planets[3] = 'Kepler421b'
print(planets)

# Rename more than 1 of them
planets[:3] = ['Mur', 'Vee', 'Ur']
print(planets)

# Give them back their old names
planets[:4] = ['Mercury', 'Venus', 'Earth', 'Mars',]

# How many planets are there?
number_of_planets = len(planets)
print(number_of_planets)

# Alphabetical order
print(sorted(planets))

# Do stuff with the primes list
print('--------------------------------------------------')
primes = [2, 3, 5, 7]
print(sum(primes))
print(max(primes))

# Real and imaginary numbers
print('--------------------------------------------------')
x = 12
# x is a real number, so its imaginary part is 0.
print(x.imag)

# This doesn't work
print(x.bit_length)

# This does
print(x.bit_length())
help(x.bit_length)

# Here's how to make a complex number, in case you've ever been curious:
c = 12 + 3j
print(c.imag)

# List methods
print('--------------------------------------------------')
# Pluto is a planet darn it!
planets.append('Pluto')
help(planets.append)
print(planets)

# Get last element of the list
print(planets.pop())

# Get any element of the list
print(planets.index('Earth'))

# Is Earth a planet?
print("Earth" in planets)

# Is Calbefraques a planet?
print("Calbefraques" in planets)

#-----------------------------------Tuples-------------------------------------
print('-----------------------------------Tuples-------------------------------------')

# Note: tuples cannot be modified like lists
t = (1, 2, 3)
print(t)
t = 1, 2, 3 # equivalent to above
print(t)

x = 0.125
print(x.as_integer_ratio())

numerator, denominator = x.as_integer_ratio()
print(numerator / denominator)

#---------------------------Stupid Python Trick(TM)----------------------------
print('---------------------------Stupid Python Trick(TM)----------------------------')
a = 1
b = 0
a, b = b, a
print(a, b)

#-----------------------------------Loops--------------------------------------
print('-----------------------------------Loops--------------------------------------')

# Print planet list
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
for planet in planets:
    print(planet, end=' ') # print all on same line
print() # This is so the function output will go to the next line

# Loop through entries in a tuple
print('--------------------------------------------------')
multiplicands = (2, 2, 2, 3, 3, 5)
product = 1
for mult in multiplicands:
    product = product * mult
print(product)

# Loop through strings
print('--------------------------------------------------')
s = 'steganograpHy is the practicE of conceaLing a file, message, \
    image, or video within another fiLe, message, image, Or video.'
msg = ''
# print all the uppercase letters in s, one at a time
for char in s:
    if char.isupper():
        print(char, end='')
print() # This is so the function output will go to the next line

# Range function
print('--------------------------------------------------')
for i in range(5):
    print("Doing important work. i =", i)

# While loops
i = 0
while i < 10:
    print(i, end=' ')
    i += 1
print() # This is so the function output will go to the next line
    
#-----------------------------List Comprehensions------------------------------
print('-----------------------------List Comprehensions------------------------------')

# Example of list comprehension
squares = [n**2 for n in range(10)]
print(squares)

# Or you could do it without one
squares = [] # pre-allocate array
for n in range(10):
    squares.append(n**2)
print(squares)

# Add an if condition
print('--------------------------------------------------')
short_planets = [planet for planet in planets if len(planet) < 6]
print(short_planets)

# str.upper() returns an all-caps version of a string
loud_short_planets = [planet.upper() + '!' for planet in planets if len(planet) < 6]
print(loud_short_planets)

# You can also split this up into multiple lines to read your code easier
[
    planet.upper() + '!' 
    for planet in planets 
    if len(planet) < 6
]

# Make list comprehensions that don't actually involve the loop variable?
print('--------------------------------------------------')
print([32 for planet in planets])

# Example of function vs. loop extension
print('--------------------------------------------------')
def count_negatives(nums):
    """Return the number of negative numbers in the given list.
    
    >>> count_negatives([5, -1, -2, 0, 3])
    2
    """
    n_negative = 0
    for num in nums:
        if num < 0:
            n_negative = n_negative + 1
    return n_negative
print(count_negatives([5, -1, -2, 0, 3]))

def count_negatives(nums):
    return len([num for num in nums if num < 0])
print(count_negatives([5, -1, -2, 0, 3]))

def count_negatives(nums):
    # Reminder: in the "booleans and conditionals" exercises, we learned about a quirk of 
    # Python where it calculates something like True + True + False + True to be equal to 3.
    return sum([num < 0 for num in nums])
print(count_negatives([5, -1, -2, 0, 3]))

#----------------------------------Strings-------------------------------------
print('----------------------------------Strings-------------------------------------')

# Single and double quotes do the same thing to create strings
x = 'Pluto is a planet'
y = "Pluto is a planet"
print(x == y)

# If you want double quotes in your string, wrap them in single quotes (or vice versa)
print("Pluto's a planet!")
print('My dog is named "Pluto"')

# Use \n to print to the next line
print('--------------------------------------------------')
hello = "hello\nworld"
print(hello)

# You can also go to a new line by pressing Enter
triplequoted_hello = """hello
world"""
print(triplequoted_hello)
triplequoted_hello == hello

# Strings are sequences, like the lists
print('--------------------------------------------------')
planet = 'Pluto'
print(planet[0])    # Indexing
print(planets[-3:]) # Slicing
print(len(planet))

# We can also loop
print('--------------------------------------------------')
print([char+'! ' for char in planet])

# String methods
print('--------------------------------------------------')
claim = "Pluto is a planet!"
print(claim.upper())            # All caps
print(claim.lower())            # All lowercase
print(claim.index('plan'))      # Search for first index of a substring
print(claim.startswith(planet)) # Starts with?
print(claim.endswith(planet))   # Ends with?

# Going between strings and lists
print('--------------------------------------------------')
words = claim.split()
print(words)

# You can also choose delimiters
print('--------------------------------------------------')
datestr = '1956-01-31'
year, month, day = datestr.split('-')
print(year, ' ', month, ' ', day)

# Or put them back in
print('--------------------------------------------------')
slashes = '/'.join([month, day, year])
print(slashes)

# We can also put in unicode characters
print('--------------------------------------------------')
unicodes = ' 👏 '.join([word.upper() for word in words])
print(unicodes)

# Building strings
print('--------------------------------------------------')
print(planet + ', we miss you.')
position = 9
print(planet + ", you'll always be the " + str(position) + "th planet to me.")

# Alternatively...
print("{}, you'll always be the {}th planet to me.".format(planet, position))

# More info on Pluto
print('--------------------------------------------------')
pluto_mass = 1.303 * 10**22
earth_mass = 5.9722 * 10**24
population = 52910390
print(
#         2 decimal points   3 decimal points, format as percent     separate with commas
"{} weighs about {:.2} kilograms ({:.3%} of Earth's mass). It is home to {:,} Plutonians.".format(
    planet, pluto_mass, pluto_mass / earth_mass, population,
)
)

# Referring to format() arguments by index, starting from 0
print('--------------------------------------------------')
s = """Pluto's a {0}.
No, it's a {1}.
{0}!
{1}!""".format('planet', 'dwarf planet')
print(s)

#-------------------------------Dictionaries-----------------------------------
print('-------------------------------Dictionaries-----------------------------------')

# Dictionaries are data structures for mapping keys to values
numbers = {'one':1, 'two':2, 'three':3}

# one, two, three are the keys, and 1, 2, 3 are the values
print(numbers['one'])

# Add another key/value pair
print('--------------------------------------------------')
numbers['eleven'] = 11
print(numbers)

# Change the values associated with a key
numbers['one'] = 'Pluto'
numbers

# Dictionary comprehensions
print('--------------------------------------------------')
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
planet_to_initial = {planet: planet[0] for planet in planets}
print(planet_to_initial)

# in tells us if something is in there
print('--------------------------------------------------')
print('Saturn' in planet_to_initial)
print('Betelgeuse' in planet_to_initial)
for k in numbers:
    print("{} = {}".format(k, numbers[k]))
    
# Get all the initials, sort them alphabetically, and put them in a space-separated string.
print('--------------------------------------------------')
print(' '.join(sorted(planet_to_initial.values())))

# dict.items()
print('--------------------------------------------------')
for planet, initial in planet_to_initial.items():
    print("{} begins with \"{}\"".format(planet.rjust(10), initial))
print('--------------------------------------------------')
help(dict)    
