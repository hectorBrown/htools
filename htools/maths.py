# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 18:06:21 2020

@author: Hector
"""

import numpy as np, scipy.integrate as integrate
from inspect import signature

#vectors
def mag(v):
    return (np.array(v)**2).sum()**0.5
def unit(v):
    return np.array(v) / mag(v) if not mag(v) == 0 else np.zeros(len(v))
def perp2d(v):
    return np.array([v[1], -v[0]])
def rotate2d(v, the):
    v = np.transpose(np.atleast_2d(np.array(v)))
    mat = np.array([[np.cos(the), np.sin(the)], [-np.sin(the), np.cos(the)]])
    return mult(mat,v).flatten()
    
def decompose(v):
    return tuple(mag(v), unit(v))
def diff_vect(v, x):
    return [derivative(v[0], x), derivative(v[1], x), derivative(v[2], x)]
def grad(f, a):
    res = []
    for i in range(len(signature(f).parameters)):
        res.append(partial_deriv(f, i + 1, a))
    return res
def div(v, a):
    res = 0
    for i in range(len(v)):
        res += partial_deriv(v[i], i + 1, a)
    return res
def curl(v, a):
    res = []
    res.append(partial_deriv(v[2], 2, a) - partial_deriv(v[1], 3, a))
    res.append(partial_deriv(v[0], 3, a) - partial_deriv(v[2], 1, a))
    res.append(partial_deriv(v[1], 1, a) - partial_deriv(v[0], 2, a))
    return res

#matrices
#matrices are defined as A[i][j] ie. A[row][col]
def det(A):
    if len(A) == 1:
        return A[0][0]
    else:
        result = 0
        for j,a in enumerate(A[0]):
            result += (-1)**j * a * det(minor(A,0,j))
        return result
def det_pretty(A, h=6):
    print("Calculating determinant of\n" + disp_mat(A, h))
    if len(A) == 1:
        return A[0][0]
    else:
        result = 0
        print("=")
        for j,a in enumerate(A[0]):
            print(str((-1)**j) + " *\n" + disp_mat(minor(A,0,j), h))
        for j,a in enumerate(A[0]):
            result += (-1)**j * a * det_pretty(minor(A,0,j), h)
        return result
def quick_det(A):
    A = list(A)
    for i, row in enumerate(A):
        A[i] = list(row)
    for i,row in enumerate(A):
        for j,a in enumerate(row):
            A[i][j] = float(a)
    if len(A) == 1:
        return A[0][0]
    else:
        factor = 1
        if A[0][0] == 0:
            if __hasnonzero(A[0]) != -1:
                __swapcols(A,0,__hasnonzero(A[0]))
            else:
                return 0
            factor = -1
        for j, a in enumerate(A[0]):
            if j == 0:
                for i in range(0, len(A)):
                    if i != 0:
                        A[i][j] /= A[0][0]
            else:
                for i in range(0, len(A)):
                    if i != 0:
                        A[i][j] -= A[0][j] * A[i][0]
        return factor * A[0][0] * quick_det(minor(A,0,0))
def quick_det_pretty(A, h=6, __factor_stack=[]):
    A = list(A)
    for i, row in enumerate(A):
        A[i] = list(row)
    for i,row in enumerate(A):
        for j,a in enumerate(row):
            A[i][j] = float(a)
    factor = 1
    for fact in __factor_stack:
        factor *= fact
    print("Factor: " + str(factor))
    print("Calculating determinant of\n" + disp_mat(A,h))
    if len(A) == 1:
        return A[0][0]
    else:
        factor = 1
        if A[0][0] == 0:
            if __hasnonzero(A[0]) != -1:
                __swapcols(A,0,__hasnonzero(A[0]))
                print("Swapping cols to give\n" + disp_mat(A,h))
            else:
                return 0
            factor = -1
        for j, a in enumerate(A[0]):
            if j == 0:
                for i in range(0, len(A)):
                    if i != 0:
                        A[i][j] /= A[0][0]
                print_A = A.copy()
                print_A[0] = A[0].copy()
                print_A[0][0] = 1
                print("Dividing through first row gives\n" + disp_mat(print_A,h))
            else:
                for i in range(0, len(A)):
                    if i != 0:
                        A[i][j] -= A[0][j] * A[i][0]
        print_A = A.copy()
        print_A[0] = A[0].copy()
        print_A[0][0] = 1
        for j in range(1, len(print_A[0])):
            print_A[0][j] = 0
        print("Clearing top row gives\n" + disp_mat(print_A,h))
        __factor_stack.append(factor * A[0][0])
        return factor * A[0][0] * quick_det_pretty(minor(A,0,0),h=h, __factor_stack=__factor_stack)
def minor(A,i,j):
    result = []
    past = False
    for row in range(0, len(A)):
        if row != i:
            result.append([])
            for col,a in enumerate(A[row]):
                if col != j:
                    if not past:
                        result[row].append(a)
                    else:
                        result[row - 1].append(a)
        else:
            past = True
    return result
def disp_mat(A, h=6):
    result = ""
    for row in A:
        for a in row:
            result += " | "
            if len(str(a)) > h:
                result += str(a)[:h]
            else:
                result += str(a)
                for i in range(0, h - len(str(a))):
                    result += " "
        result += " |\n"
    return result
def iden_mat(n):
    res = []
    for row in range(0,n):
        res.append([])
        for col in range(0, n):
            res[row].append(1 if row == col else 0)
    return res
def __hasnonzero(li):
    for i, num in enumerate(li):
        if num != 0:
            return i
    return -1
def __swapcols(A,j1,j2):
    for i in range(0, len(A)):
        A[i][j1],A[i][j2] = A[i][j2],A[i][j1]
def solve_simul(A):
    delt = quick_det([row[:-1] for row in A])
    res = []
    for i in range(len(A)):
        res.append(__deltn(A,i) / delt)
    return res
def cofactors(n):
    res = []
    for row in range(n):
        res.append([])
        for col in range(n):
            res[row].append((-1)**(row + col))
    return res
    
def __deltn(A, n):
    return quick_det([row[:n - 2] + b + row[n + 1:] for row,b in zip([row[:-1] for row in A], [row[-1:] for row in A])])
def mult(A, B):
    A = np.array(A)
    B = np.array(B)
    res = np.zeros((len(A), len(np.transpose(B))))
    if len(np.transpose(A)) == len(B):
        for r,row in enumerate(A):
            for c, col in enumerate(np.transpose(B)):
                res[r][c] = sum([x * y for x,y in zip(row,col)])
    return res

#calculus
def derivative(f, a, h=0.01):
    return (f(a+h) - f(a-h))/(2*h)
def derivative_fun(f, n, h=0.01):
    class deriv:
        def __init__(self, n, fs):
            self.n = n
            self.fs = fs
        def df(self, x):
            return derivative(self.fs[self.n - 1], x)
    fs = [f]
    for i in range(1,n + 1):
        df = deriv(i, fs)
        fs.append(df.df)
    return fs[len(fs) - 1]
def hintegrate(f, a, b, polar=False):
    if polar:
        return integrate.quad(lambda x: 0.5 * f(x)**2, a, b)[0]
    else:
        return integrate.quad(f, a, b)[0]
#FIXME: THIS OUGHT TO BE ABLE TO ACCEPT LISTS
def hintegrate_fun(f,c,n):
    class inte:
        def __init__(self, n, fs, cs):
            self.n = n
            self.fs = fs
            self.cs = cs if type(cs) is list or type(cs) is np.ndarray else [cs]
        def intf(self, x):
            return hintegrate(self.fs[self.n - 1], 0, x) + self.cs[self.n - 1] 
    fs = [f]
    for i in range(1,n + 1):
        df = inte(i, fs, c)
        fs.append(df.intf)
    return fs[len(fs) - 1]
def partial_deriv(f, n, a, h=0.01):
    params = len(signature(f).parameters)
    if n > 1 and n < params:
        return derivative(lambda x: f(*a[:n-1],x,*a[n:]), a[n-1])
    elif n ==1:
        return derivative(lambda x: f(x, *a[n:]), a[n-1])
    else:
        return derivative(lambda x: f(*a[:n-1],x), a[n-1])
def vol_rev(a, b, y):
    return np.pi * integrate.quad(lambda x: y(x)**2, a, b)[0]
def surface_rev(a, b, y):
    return 2 * np.pi * integrate.quad(lambda x: y(x) * (1 + derivative(y, x)**2)**0.5, a, b)[0]
def arc_length(f1, a, b, f2 = None, polar = False):
    if polar:
        return integrate.quad(lambda the: np.sqrt(f1(the)**2 + derivative(f1,the)**2), a, b)[0]
    else:
        if f2 is None:
            return integrate.quad(lambda x: np.sqrt(1 + derivative(f1, x)**2), a, b)[0]
        else:
            return integrate.quad(lambda x: np.sqrt(derivative(f1, x)**2 + derivative(f2, x)**2), a, b)[0]
def lhopital(f, g, t):
    counter = 1
    dg = g
    while round(dg(t),6) == 0:
        df = derivative_fun(f, counter)
        dg = derivative_fun(g, counter)
        counter += 1
    return df(t) / dg(t)
def mean(f, a, b):
    return (b-a)**-1 * integrate.quad(f, a, b)[0]
def RMS(f, a, b):
    return ((b-a)**-1 * integrate.quad(lambda x: f(x)**2, a, b)[0])**0.5

#complex
def comp_mag(z):
    return (np.real(z)**2 + np.imag(z)**2)**0.5
def square_to_polar(z):
    return tuple((mag(z), np.angle(z)))
def polar_to_square(z):
    return complex(z[0] * np.cos(z[1]), z[0] * np.sin(z[1]))

#series
#FIXME:
# def taylor(f,a,n):
#     funcs = []
#     for i in range(n):
#         if i == 0:
#             funcs.append(lambda x: f(a))
#         else:
#             funcs.append(lambda x: derivative_fun(f,i)(a)/np.math.factorial(i) * (x-a)**i)
#     print([f(1) for f in funcs])
#     return lambda x: sum([f(x) for f in funcs])
def fibon_lazy():
    a = 1
    yield a
    b = 1
    yield b
    while True:
        c = a + b
        yield c
        a = b
        b = c
def fibon(n, _list=True):
    a = 1
    b = 1
    if _list: res = [1,1]
    if n > 2:
        for i in range(n - 2):
            c = a + b
            if _list: res.append(c)
            a = b
            b = c
    if n == 1 and _list: return [1]
    if n == 1 or n == 2 and not _list: return 1
    return res if _list else c