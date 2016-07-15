# function x^3 - 6x^2 + 11x -6 

x_old = 0
x_new = 0.5
gamma = 0.01
precision = 1e-5

def f_derivative(x):
	return (3*x**2 - 12*x + 11)

while abs(x_new - x_old) > precision:
	x_old = x_new
	x_new = x_old +gamma*f_derivative(x_old)
	print f_derivative(x_old),x_new
