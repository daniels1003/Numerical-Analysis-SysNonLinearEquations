import numpy as np 
import os
import xlrd
import xlwt 
import openpyxl 
from collections import defaultdict
import operator
import sys
import scipy
from scipy import optimize
from scipy.optimize import fsolve
print(sys.version)
print(sys.executable)
from scipy import stats 
from scipy import misc
import pandas as pd 
from pandas import ExcelWriter
import matplotlib
import matplotlib.pyplot as plt
import sympy as sp
from sympy.abc import x
from sympy import *
from sympy import Symbol, Derivative, utilities
from sympy.utilities.lambdify import lambdastr
import math 
import symengine
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from textwrap import wrap

def f(x):
    return np.sin(x)

def dfdx(x):
    return np.cos(x)

def Newton(f, dfdx, num_iterations, starting_x,problem):
	#starting_x = 4
	
	if problem == 1: 
		count = 0
		x = starting_x
		print("Starting Newtons Iteration Method with x1 = ",starting_x, " on the interval [3,4].")
		while (count < num_iterations):
			error = abs(f(x))
			print("The error on Newton Method iteration for iteration number ", count," out of ", num_iterations," interations has error of : ", error, " from the zero of the function, 3.14159.")

			x = x - float(f(x))/dfdx(x)
			count += 1

		print("The final approximation gave the value of ", x, " with error of ", error, " from the zero of the function ", np.pi)
		print("The apparent order of convergence is: 2")
		return 0
	
	iterations = ['x1','x2','x3','x4','x5','x6','x7','x8']
	count = 0; problem2IterationsList = []; problem2Pairs = np.zeros((4,2))
	x = starting_x
	print("x is starting at:",x)
	if problem == 2:
		problem2IterationsList.append(x)
	print("Starting Newtons Iteration Method with x1 = ",starting_x)
	while (count < num_iterations):
		error = abs(f(x))
		print("The error on Newton Method iteration for iteration number ", count," out of ", num_iterations," interations has error of : ", error, " from the zero of the function.")

		x = x - float(f(x))/dfdx(x)
		if problem == 2:
			problem2IterationsList.append(x)
		count += 1
	for i in range(4): 
		problem2Pairs[i][0] = problem2IterationsList[i]
		problem2Pairs[i][1] = problem2IterationsList[i+4]
	for i in range(4): 
		print("Pair of iterations, ",iterations[i],":",problem2IterationsList[i]," and ",iterations[i+4],":",problem2IterationsList[i+4],".")
	#print("The final approximation gave the value of ", x, " with error of ", error, " from the zero of the function. ")
	print("From analyzing the pairs of x-iterates we can conclude that newtons method starts to diverge and cannot always be relied upon.")
	return 0

#Problem 1 The function f(x) = sin(x) has a zero on the interval (3,4),
# namely, x∗ = π. Perform three iterations of Newton method to 
# approximate this zero, using x1 = 4. Determine the absolute error 
# in each of the computed approximations. What is the apparent order of convergence?
	 
		

		
def Problem_1(f,dfdx,num_iterations,starting_x):
	#f = np.sin(x); dfdx = np.cos(x)
	problem = 1
	Newton(f, dfdx, num_iterations,starting_x,problem)


print("Starting problem number 1...")
Problem_1(f,dfdx,3,4)
print("\n")


def f2(x):
	return x**3 - x - 3

def df2dx2(x):
	return 3*x**2 - 1 

# Problem 2 Apply the Newton’s method to ﬁnd the solution 
# to x3 −x−3 = 0 starting with x1 = 0. Compute x2,x3,x4,x5,x6,x7 
# and x7 and compare pair of numbers (x1,x5), (x2,x6), (x3,x7) and
# (x4,x8). What can you conclude from this computations (Use your computer code)


def Problem_2(f2,df2dx2,num_iterations,starting_x):
	problem = 2
	Newton(f2,df2dx2,num_iterations,starting_x,problem)

#Newton(f2,df2dx2,8,0,2)
print("\n")
print("Starting problem number 2 using newtons method...")
Problem_2(f2,df2dx2,8,0)
print("\n")

def f3(x): 
	return (math.exp(x) - x**2 + 3*x - 2)


"""
Problem 3 Find an approximation by the method of false position 
for the root of function f(x) = ex −x2 + 3x−2 accurate to within 
10−5 (absolute error) on the interval [0,1]. (Use your computer code)
"""
interval = [0,1]; tolerance = 10**-5

def False_Position(interval,tolerance):
	error = 1; counter = 0; a = interval[0]; b = interval[1]; first_x = 0
	if f3(a) * f3(b) >= 0:
		print("You have not used the correct a and b from the interval.")
		return -1
	#print(f3(0.25753))

	x = a - ((b-a)/(f3(b)-f3(a)))*f3(a)
	first_x = x
	error = abs(f3(x))

	while(error > tolerance and counter <= 1001):

		x = a - ((b-a)/(f3(b)-f3(a)))*f3(a)
		error = abs(f3(x))
		counter += 1
		if f3(x) == 0:
			return x
		elif f3(a)*f3(x) < 0: 
			b = x
		else:
			a = x
		
	print("The number of iterations reached until the tolerance for error has been met in: ", counter," iterations, with error of: ", error, " and tolerance of ", tolerance)
	print("The value of the function at our last x is: ", f3(x), " with x being: ", x)

print("\n")
print("Starting problem number 3 using the method of False Position...")
False_Position(interval,tolerance)
print("\n")
#exit(1)
"""


"""
# Problem 4 Find an approximation to √3 correct to within 10−4 using the 
# Bisection method (Hint: Consider f(x) = x2 −3.) (Use your computer code)

def f4(x):
	return x**2 - 3

tolerance = 10**-4; interval = [1,3]
def Bisection_Method(interval,tolerance):
	a = interval[0]; b = interval[1]; counter = 0; max_iterations = 1000
	x = (a+b)/2
	if f4(a)*f4(b) >= 0:
		print("The assumption that f(a)*f(b) < 0 is not true so the program will now terminate...") 
	for n in range(max_iterations): 
		if f4(a)*f4(x) < 0: 
			b = x
		elif f4(x)*f4(b) < 0: 
			a = x
		x = (a+b)/2; counter += 1 
		if abs(f4(x)) <= tolerance or abs(b-a) <= tolerance:
			print("The approximation of f(x)=0 for f(x)=x^2-3 is: ", x)
			print("The Bisection method required ",counter," iterations for this approximation.")
			return

print("Starting problem 4 using the Bisection Method...")
Bisection_Method(interval,tolerance)
print("The exact zero of f(x) is : ", 3**(1/2))
print("\n")
#exit(1)

# Problem 5: Consider the nonlinear equations:
def f5(x1,x2): 
	return x1*x2**2 + x1*x2 - x1**4 - 1
def g5(x1,x2):
	return x1**2 + x2 - 2
def F(x):
	x1 = x[0]; x2 = x[1]; 
	#f = Matrix((1,2))
	f = np.zeros(2)
	f[0] = f5(x1,x2)
	f[1] = g5(x1,x2)
	return f

def Jac(x_1,x_2): 
	x1,x2 = symbols('x1 x2')
	F = x1*x2**2 + x1*x2 - x1**4 - 1
	G = x1**2 + x2 - 2
	J = np.zeros((2,2))
	jacobian_list = []
	jacobian_list.append(diff(F,x1)); jacobian_list.append(diff(F,x2)); jacobian_list.append(diff(G,x1)); jacobian_list.append(diff(G,x2));
	
	for i in range(4):
		if i < 2: 
			func = jacobian_list[i]
			func = lambdify((x1,x2),func,modules='numpy')

			#print(func(0.8,0.8))
			#print(type(func))
			#exit(1)
			J[0][i] = func(x_1,x_2)
		elif i > 1:
			func = jacobian_list[i]
			func = lambdify((x1,x2),func,modules='numpy')
			J[1][i-2] = func(x_1,x_2)

	return J 



def Newton_system(tol, max_iterations):
	func_iter = np.zeros((2,42))
	x_iters = np.zeros((2,42))
	x1 = 0.8; x2 = 0.8
	x = [x1,x2]
	F_value = F(x)
	#print(F_value)

	iteration_counter = 0
	while (abs(F_value[0]) > tol and abs(F_value[1]) > tol and iteration_counter < max_iterations):
		x_iters[0][iteration_counter] = x[0]
		x_iters[1][iteration_counter] = x[1]

		func_iter[0][iteration_counter] = F_value[0]
		func_iter[1][iteration_counter] = F_value[1]

		delta = np.linalg.solve(Jac(x1,x2), -F_value)
		x = x + delta
		F_value = F(x)

		iteration_counter += 1

    # Here, either a solution is found, or too many iterations
	if abs(F_value[0]) > tol and abs(F_value[1]) > tol:
		iteration_counter = -1
	return x, iteration_counter, func_iter, x_iters

def displayProblem_5_Plots():
	x, counter, function_iterations, x_iters = Newton_system(0.00001,1000)
	print("The solution for the system of nonlinear equations is x1: ", x[0], " and x2: ", x[1], " after ", counter," iterations.")
	#print(f5(x[0],x[1]))
	#print(g5(x[0],x[1]))

	xf = x_iters[0]; xg = x_iters[1]
	x1 = arange(0.0,1.1,0.005)
	x2 = arange(0.0,1.1,0.005)
	X,Y = meshgrid(x1, x2) # grid of point
	Z = f5(X, Y) # evaluation of the function on the grid
	Z2 = g5(X,Y)
	Z3 = function_iterations[0]
	Z4 = function_iterations[1]
	fig = plt.figure()
	#plt.figtext(0.8,0.8,'')
	ax = fig.gca(projection='3d')
	f_x = "f(x1,x2) = x1*x2**2 + x1*x2 - x1**4 - 1"
	g_x = "g(x1,x2) = x1**2 + x2 - 2"

	ax.set_title("Newtons Method for solution of a pair of nonlinear functions of two variables: x1, x2\n%s" % "\n".join(wrap("f(x1,x2) = x1*x2^2 + x1*x2 - x1^4 - 1 = 0\n,\n%s", width=60)) % "\n".join(wrap("g(x1,x2) = x1^2 + x2 - 2 = 0", width=80)))

	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap="Reds",linewidth=0, antialiased=False)
	#surf.ax.set_title("f(x1,x2) = x1*x2**2 + x1*x2 - x1**4 - 1")
	surf2 = ax.plot_surface(X,Y,Z2, rstride=1,cstride=1,cmap="Blues",linewidth=0,antialiased=False)
	plt.plot(xf,xg,Z3,"go")
	ax.scatter3D(xf[40],xg[40],Z4[40],color="black")
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	ax.set_xlabel('x1')
	ax.set_ylabel('x2')
	ax.set_zlabel('z')
	ax.view_init(0,90)
	ax.text(0.8,0.8,Z4[0],"Starting Iteration for f")
	ax.text(0.8,0.8,Z3[0],"Starting Iteration for g")
	ax.text(xf[40],xg[40],Z4[40],"Ending Iteration for f")
	#ax.text(0.8,0.8,Z3[41],"Ending Iteration for g")
	F_colorBar = fig.colorbar(surf, shrink=0.5, aspect=5)
	F_colorBar.ax.set_title("F(x1,x2)")
	G_colorBar = fig.colorbar(surf2,shrink=0.5,aspect=5)
	G_colorBar.ax.set_title("G(x1,x2)")
	#plt.annotate('Staring Iteration (0.8,0.8) for f(x1,x2)', xy=(x_iters[0][0], x_iters[1][0]), xytext=(0.5,0.2), arrowprops=dict(facecolor='black', shrink=0.5),)
	#plt.annotate('Staring Iteration (0.8,0.8) for g(x1,x2)', xy=(x_iters[0][0], x_iters[1][0]), xytext=(0.5,0.4), arrowprops=dict(facecolor='black', shrink=0.01),)
	#plt.annotate('Final Iteration for  both functions.', xy=(x_iters[0][41], x_iters[1][41]), xytext=(0.5,0.6), arrowprops=dict(facecolor='black', shrink=0.05),)
	plt.show()


	xf = x_iters[0]; xg = x_iters[1]
	#x1 = arange(0.0,1.1,0.005)
	#x2 = arange(0.0,1.1,0.005)
	X,Y = meshgrid(xf, xg) # grid of point
	Z = f5(X, Y) # evaluation of the function on the grid
	Z2 = g5(X,Y)
	Z3 = function_iterations[0]
	Z4 = function_iterations[1]
	fig = plt.figure()
	#plt.figtext(0.8,0.8,'')
	ax = fig.gca(projection='3d')
	f_x = "f(x1,x2) = x1*x2**2 + x1*x2 - x1**4 - 1"
	g_x = "g(x1,x2) = x1**2 + x2 - 2"
	#ax.set_title("\n".join(wrap('Newtons Method for solution of a pair of nonlinear functions of two variables: x1, x2, f(x1,x2) = x1*x2**2 + x1*x2 - x1**4 - 1, g(x1,x2) = x1**2 + x2 - 2',50)))
	ax.set_title("Newtons Method for solution of a pair of nonlinear functions of two variables: x1, x2\n%s" % "\n".join(wrap("f(x1,x2) = x1*x2^2 + x1*x2 - x1^4 - 1\n%s", width=39)) % "\n".join(wrap("g(x1,x2) = x1^2 + x2 - 2", width=60)))
	#ax.set_title('f(x1,x2) = x1*x2**2 + x1*x2 - x1**4 - 1')
	#ax.set_title('g(x1,x2) = x1**2 + x2 - 2')
	plt.plot(xf,xg,Z3,"ro")
	plt.plot(xf,xg,Z4,"bo")
	ax.scatter3D(xf[40],xg[40],Z4[40],color="black")
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	ax.set_xlabel('x1')
	ax.set_ylabel('x2')
	ax.set_zlabel('z')
	ax.view_init(0,90)
	ax.text(0.8,0.8,Z4[0],"  Starting Iteration for f")
	ax.text(0.8,0.8,Z3[0],"  Starting Iteration for g")
	ax.text(xf[40],xg[40],Z4[40],"  Ending Iteration for f and g at point x1: " +str(x[0])+" and x2: " + str(x[1]))
	#ax.text(0.8,0.8,Z3[41],"Ending Iteration for g")
	#F_colorBar = fig.colorbar(surf, shrink=0.5, aspect=5)
	#F_colorBar.ax.set_title("F(x1,x2)")
	#G_colorBar = fig.colorbar(surf2,shrink=0.5,aspect=5)
	#G_colorBar.ax.set_title("G(x1,x2)")
	#plt.annotate('Staring Iteration (0.8,0.8) for f(x1,x2)', xy=(x_iters[0][0], x_iters[1][0]), xytext=(0.5,0.2), arrowprops=dict(facecolor='black', shrink=0.5),)
	#plt.annotate('Staring Iteration (0.8,0.8) for g(x1,x2)', xy=(x_iters[0][0], x_iters[1][0]), xytext=(0.5,0.4), arrowprops=dict(facecolor='black', shrink=0.01),)
	#plt.annotate('Final Iteration for  both functions.', xy=(x_iters[0][41], x_iters[1][41]), xytext=(0.5,0.6), arrowprops=dict(facecolor='black', shrink=0.05),)
	plt.show()
	exit(1)

print("Starting problem 5...")
displayProblem_5_Plots()



