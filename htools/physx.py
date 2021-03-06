# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 19:27:13 2020

@author: Hector
"""

import numpy as np

G = 6.6738480e-11
M_sun = 1.989e30
M_earth = 5.972e24
R_earth = 6371e3

def rocket_vel(v_0, u, M_t, M_0):
    return v_0 - u * np.log(M_t/M_0)
def CoM_to_lab(p_ast, q_ast, the_ast, m_1, m_2):
    p_ast = np.array(p_ast)
    q_ast = np.array(q_ast)
    return { "p_1": (m_1/m_2 + 1) * p_ast, "p_2": 0, "q_1": q_ast + (m_1/m_2) * p_ast, "q_2": p_ast - q_ast, "R_dot": p_ast / m_2, "the": np.atan(np.sin(the_ast)/((m_1/m_2) + np.cos(the_ast))), "phi": (1/2) * (np.pi - the_ast)}
def lab_to_CoM(r_1,r_2, m_1, m_2, r_dot):
    r_1 = np.array(r_1)
    r_2 = np.array(r_2)
    R = CoM_R(r_1, r_2, m_1, m_2)
    r_dot = np.array(r_dot)
    return { "r_1_ast": r_1 - R, "r_2_ast": r_2 - R, "p_ast": -reduced_mass(m_1, m_2) * r_dot}
def CoM_R(r_1, r_2, m_1, m_2):
    r_1 = np.array(r_1)
    r_2 = np.array(r_2)
    return (m_1 * r_1 + m_2 * r_2)/(m_1 + m_2)
def reduced_mass(m_1, m_2):
    return (m_1 * m_2)/(m_1 + m_2)
def SHM_a(f,x,ang=False):
    if ang:
        return -(f**2)*x
    else:
        return -((2 * np.pi * f)**2) * x
def SHM_v(f,A,x,ang=False):
    if ang:
        return f * np.sqrt(A**2 - x**2)
    else:
        return 2 * np.pi * f * np.sqrt(A**2 - x**2)
def SHM_x(f,A,t,ang=False):
    if ang:
        return A * np.cos(f * t)
    else:
        return A * np.cos(2 * np.pi * f * t)
def KIII_T(M, a):
    return np.sqrt((4 * np.pi**2 * a**3)/(G * M))
def KIII_a(M, T):
    return ((T**2 * G * M)/(4 * np.pi**2))**(1/3)
def E_orbit(M, m, a):
    return -(G * M * m)/ (2 * a)
def E_spring(m, ome, A):
    return 0.5 * m * ome**2 * A**2
def K_spring(m, ome, A, t, phi):
    return 0.5 * m * ome**2 * A**2 * np.sin(ome * t + phi)**2
def U_spring(m, ome, A, t, phi):
    return 0.5 * m * ome**2 * A**2 * np.cos(ome * t + phi)**2
#signals
def sin(f, A, phi=0):
    return lambda x: (A / 2) * np.sin(2 * np.pi * f * x + phi)
def square(f,A, phi=0):
    def func(x):
        while x > (1/f):
            x -= (1/f)
        if x < (1/f) / 2:
            return (A/2)
        else:
            return -(A/2)
    def iter_x(x):
        try:
            iter(x)
            return np.array([func(i + phi) for i in x])
        except TypeError:
            return func(x + phi)
    return iter_x
def triangle(f,A, phi=0):
    def func(x):
        while x > (1/f):
            x -= (1/f)
        if x < (1/f) / 4:
            return 2 * A * f * x
        elif x < (3/f) / 4:
            return -2 * A * f * x + A
        else:
            return 2 * A * f * x - 2 * A
    def iter_x(x):
        try:
            iter(x)
            return np.array([func(i + phi) for i in x])
        except TypeError:
            return func(x + phi)
    return iter_x