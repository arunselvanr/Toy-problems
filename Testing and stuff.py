class complex(object):
    def __init__(self, a,b):
        self.real = float(a)
        self.complex = float(b)

def complex_sum(x1, x2):
    sum = complex(0.0,0.0)
    sum.real = float(x1.real) + float(x2.real)
    sum.complex = float(x1.complex) + float(x2.complex)
    return(sum)

def complex_prod(x1,x2):
    prod=complex(0,0)
    prod.real = x1.real * x2.real - x1.complex * x2.complex
    prod.complex = x1.real * x2.complex + x1.complex * x2.real
    return(prod)

y1 = complex(4.5,6.98)
y2 = complex(3.45,7.89)
print('(%f + %f i )+ (%f + %f i) = %f + %f i' %(y1.real,y1.complex,y2.real,y2.complex, complex_sum(y1, y2).real, complex_sum(y1, y2).complex))

print('(%f + %f i ) x (%f + %f i) = %f + %f i' %(y1.real,y1.complex,y2.real,y2.complex, complex_prod(y1, y2).real, complex_prod(y1, y2).complex))

b = input('Enter any number')
print(type(b))
print('Square of the number is %d' %float(b*b))
#################################################################################################
#################################################################################################
#################################################################################################
