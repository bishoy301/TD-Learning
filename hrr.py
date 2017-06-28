#!/usr/bin/env python

import numpy

def hrr(length,normalized=False):
    "Creates Holographic Reduced Representations"

    if normalized:
        x = numpy.random.uniform(-numpy.pi,numpy.pi,(length-1)/2)
        if length % 2:
            x = numpy.real(numpy.fft.ifft(numpy.concatenate([numpy.ones(1),numpy.exp(1j*x),numpy.exp(-1j*x[::-1])])))
        else:
            x = numpy.real(numpy.fft.ifft(numpy.concatenate([numpy.ones(1),numpy.exp(1j*x),numpy.ones(1),numpy.exp(-1j*x[::-1])])))
    else:
        x = numpy.random.normal(0.0,1.0/numpy.sqrt(length),length)
        
    return x

def hrrs(length,n=1,normalized=False):
    "Creates a matrix of n HRRs, one per row."
    return numpy.row_stack([hrr(length,normalized) for x in range(n)])

def inv(x):
    "Computes the -exact- inverse of an HRR."
    x = numpy.fft.fft(x)
    return numpy.real(numpy.fft.ifft((1.0/numpy.abs(x))*numpy.exp(-1j*numpy.angle(x))))

def pinv(x):
    "Computes the pseudo-inverse of an HRR."
    return numpy.real(numpy.fft.ifft(numpy.conj(numpy.fft.fft(x))))

def convolve(x,y):
    "Computes the circular convolution of two HRRs."
    return numpy.real(numpy.fft.ifft(numpy.fft.fft(x)*numpy.fft.fft(y)))

def mconvolve(x):
    "Computes the circular convolution of a matrix of HRRs."
    return numpy.real(numpy.fft.ifft(numpy.apply_along_axis(numpy.prod,0,numpy.apply_along_axis(numpy.fft.fft,1,x))))
    
def oconvolve(x,y):
    "Computes the convolution of all pairs o HRRs in the x and y matrices."
    return numpy.row_stack([ [convolve(x[i,:],y[j,:]) for i in range(x.shape[0]) ] for j in range(y.shape[0]) ])

def correlate(x,y,invf=inv):
    "Computes the correlation of the two provided HRRs."
    return convolve(x,invf(y))

def compose(x,y):
    "Composes two HRRs using addition in angle space."
    x = numpy.fft.fft(x)
    y = numpy.fft.fft(y)
    return numpy.real(numpy.fft.ifft((numpy.abs(x+y)/2.0)*numpy.exp(1j*numpy.angle(x+y))))

def mcompose(x):
    "Composes a matrix of HRRs, one per row, using addition in angle space."
    x = numpy.apply_along_axis(numpy.fft.fft,1,x)
    return numpy.real(numpy.fft.ifft(numpy.apply_along_axis(numpy.mean,0,numpy.abs(x))*numpy.exp(1j*numpy.angle(numpy.apply_along_axis(numpy.sum,0,x)))))

def decompose(x,y):
    "Decomposes two HRRs using addition in angle space."
    return compose(x,-y)
