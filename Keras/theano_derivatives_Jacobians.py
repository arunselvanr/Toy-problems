import numpy
import theano
from theano import tensor as T
from theano import function
from theano import pp

################################### Taking the derivative wrt to x
###################################
x = T.dscalar()
y = x**2 + 3*x + 4
g_y = T.grad(y, x)
print pp(g_y)
gradient = function([x], g_y)
p = gradient(3.5)

z = T.exp(-x)
g_z = T.grad(z, x)
print "Derivative of exponential is", pp(g_z)
####################################
#################################### theano scan and looping
k = T.iscalar()
A = T.matrix()

result, updates = theano.scan(fn=lambda prior_result, A: prior_result * A,
                              outputs_info=T.ones_like(A), #One can also use T.zeros_like(A) this would have come in handy
                              non_sequences=A,             #if we had to loop for A + A + A, say k times.
                              n_steps=k)
A_k = result[-1]

power = function([A, k], outputs=A_k, updates=updates)

print power([[1, 2], [2,3]], 4)

##########################Scanning through a vector
coeff = T.dvector()
x = T.dscalar()
max_coeff = 1000
components, updates = theano.scan(fn=lambda power, coef, free_variable: coef*(free_variable**power),
                         outputs_info=None, non_sequences=x, sequences=[T.arange(max_coeff), coeff]
                         )
poly = components.sum()

calc_poly = function([coeff, x], poly)

print calc_poly([1,0,2], 3)
#Please list all the sequences-variables first, then the non-sequences. Maintain
#the same order while listing within sequences and non-sequences as well.

#########################################
#########################################Let's compute the Jacobian
x = T.dvector()
y = x**2
grad, updates = theano.scan(fn=lambda i, y, x: T.grad(y[i], x), outputs_info=None,
                                  sequences=T.arange(y.shape[0]), non_sequences=[y, x]
                                  )
find_grad = function([x], grad)
print find_grad([2,3,4])

#########################################
#########################################Let's calculate the Hessian

cost = y.sum()
gy = T.grad(cost, x)

H, updates = theano.scan(fn=lambda i, gy, x: T.grad(gy[i], x), sequences=T.arange(gy.shape[0]),
                         non_sequences=[gy, x], outputs_info=None
                         )
Hessian = function([x], H, updates=updates)
print(Hessian([1,2,3]))