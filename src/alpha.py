def alpha_function(x,N,c=0):
    return 0.0*x

def alpha_constant(x,N,c=0.5):
    return c*x

def alpha_step(x,N,step=0.5):

    if hasattr(x,'__iter__'):
        y = x.copy()
        for i,element in enumerate(y):
            y[i] = alpha_step(y[i],N,step)
        return y

    if float(x) > step*float(N):
        return 0.0
    else:
        return 1.0

def alpha_poly(_x,_N,order=1):
    x = float(_x)
    N = float(_N)
    return (1.0 - x/N)**order

def alpha_sigmoid(_x,_N,a=20,b=0.5):
    x = float(_x)
    N = float(_N)
    arg = x/N - b
    return 1.0/(1.0 + np.exp(a*arg))

def alpha_sin(_x,_N,n=5):
    x = float(_x)
    N = float(_N)
    w = 2 * np.pi * n / float(N)
    return 1.0 - np.cos(w * x) ** 2

def alpha_flip(_x,_N,p=5):
    x = float(_x)
    N = float(_N)
    if hasattr(x,'__iter__'):
        y = x.copy()
        for i in xrange(x.shape[0]):
            y[i] = f(x[i],p,N)
        return y
    else:
        if x < np.abs(p):
            if p > 0:
                return 1.0
            else:
                return 0.0
        else:
            return f(x-np.abs(p),-p,N)