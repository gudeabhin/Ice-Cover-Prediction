#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 02:48:58 2020

@author: abhingude
"""
import random
import math
#Done    
def get_dataset():
    m=[[1855, 118], [1856, 151], [1857, 121], [1858, 96], [1859, 110], [1860, 117], [1861, 132], [1862, 104], [1863, 125], [1864, 118], [1865, 125], [1866, 123], [1867, 110], [1868, 127], [1869, 131], [1870, 99], [1871, 126], [1872, 144], [1873, 136], [1874, 126], [1875, 91], [1876, 130], [1877, 62], [1878, 112], [1879, 99], [1880, 161], [1881, 78], [1882, 124], [1883, 119], [1884, 124], [1885, 128], [1886, 131], [1887, 113], [1888, 88], [1889, 75], [1890, 111], [1891, 97], [1992, 112], [1893, 101], [1894, 101], [1895, 91], [1896, 110], [1897, 100], [1898, 130], [1899, 111], [1900, 107], [1901, 105], [1902, 89], [1903, 126], [1904, 108], [1905, 97], [1906, 94], [1907, 83], [1908, 106], [1909, 98], [1910, 101], [1911, 108], [1912, 99], [1913, 88], [1914, 115], [1915, 102], [1916, 116], [1917, 115], [1918, 82], [1919, 110], [1920, 81], [1921, 96], [1922, 125], [1923, 104], [1924, 105], [1925, 124], [1926, 103], [1927, 106], [1928, 96], [1929, 107], [1930, 98], [1931, 65], [1932, 115], [1933, 91], [1934, 94], [1935, 101], [1936, 121], [1937, 105], [1938, 97], [1939, 105], [1940, 96], [1941, 82], [1942, 116], [1943, 114], [1944, 92], [1945, 98], [1946, 101], [1947, 104], [1948, 96], [1949, 109], [1950, 122], [1951, 114], [1952, 81], [1953, 85], [1954, 92], [1955, 114], [1956, 111], [1957, 95], [1958, 126], [1959, 105], [1960, 108], [1961, 117], [1962, 112], [1963, 113], [1964, 120], [1965, 65], [1966, 98], [1967, 91], [1968, 108], [1969, 113], [1970, 110], [1971, 105], [1972, 97], [1973, 105], [1974, 107], [1975, 88], [1976, 115], [1977, 123], [1978, 118], [1979, 99], [1980, 93], [1981, 96], [1982, 54], [1983, 111], [1984, 85], [1985, 107], [1986, 89], [1987, 87], [1988, 97], [1989, 93], [1990, 88], [1991, 99], [1992, 108], [1993, 94], [1994, 74], [1995, 119], [1996, 102], [1997, 47], [1998, 82], [1999, 53], [2000, 115], [2001, 21], [2002, 89], [2003, 80], [2004, 101], [2005, 95], [2006, 66], [2007, 106], [2008, 97], [2009, 87], [2010, 109], [2011, 57], [2012, 87], [2013, 117], [2014, 91], [2015, 62], [2016, 65], [2017, 94], [2018, 86], [2019, 70]]
    return m


#Done
def print_stats(dataset):
    length=len(dataset)
    count=0
    for i in dataset:
        count+=i[1]
    mean=count/length
    cnt1=0
    for j in dataset:
        p=(j[1]-mean)**2
        cnt1+=p
    vari=cnt1/(length-1)
    covar=vari**0.5
    print(length)
    print("{:.2f}".format(mean))
    print("{:.2f}".format(covar))


#Done
def regression(beta_0,beta_1):
    k=get_dataset()
    length=len(k)
    cnt=0
    for i in k:
        p=(beta_0+(beta_1*i[0])-i[1])
        p=math.pow(p,2)
        cnt+=p
    x=cnt/length
    return x

#helper func()
def regr(beta_0,beta_1,k):
    length=len(k)
    cnt=0
    for i in k:
        p=(beta_0+(beta_1*i[0])-i[1])
        p=math.pow(p,2)
        cnt+=p
    x=cnt/length
    return x


#Done
def gradient_descent(beta_0,beta_1):
    k=get_dataset()
    length=float(len(k))
    cnt1=0
    cnt2=0
    for i in k:
        a=beta_0+(beta_1*i[0])-i[1]
        b=(beta_0+(beta_1*i[0])-i[1])*i[0]
        cnt1+=a
        cnt2+=b
    par_b0=(cnt1*2)/length
    par_b1=(cnt2*2)/length
    l=(par_b0,par_b1)
    return l


#helper func()
def grad_des(beta_0,beta_1,k):
    length=len(k)
    cnt1=0
    cnt2=0
    for i in k:
        a=beta_0+(beta_1*i[0])-i[1]
        b=a*i[0]
        cnt1+=a
        cnt2+=b
    par_b0=(cnt1*2)/length
    par_b1=(cnt2*2)/length
    l=(par_b0,par_b1)
    return l

#helper func()
def grad_des1(beta_0,beta_1,k):
    r=random.randint(0,164)
    xj=k[r][0]
    yj=k[r][1]
    par_b0=2*(beta_0+beta_1*xj-yj)
    par_b1=par_b0*xj
    l=(par_b0,par_b1)
    return l
    
    
#Done
def iterate_gradient(T,eta):
    b_0=0
    b_1=0
    for i in range(1,T+1):
        x=gradient_descent(b_0,b_1)
        b0_new=b_0-eta*(x[0])
        b1_new=b_1-eta*(x[1])
        reg=regression(b0_new,b1_new)
        print(str(i)+" "+"{:.2f}".format(b0_new)+" "+"{:.2f}".format(b1_new)+" "+"{:.2f}".format(reg))
        b_0=b0_new
        b_1=b1_new


#Done      
def compute_betas():
    ds=get_dataset()
    s0=0
    s1=0
    for i in ds:
        s0+=i[0]
        s1+=i[1]
    mean0=s0/len(ds)
    mean1=s1/len(ds)
    cnt1=0
    cnt2=0
    for i in ds:
        x1=i[0]-mean0
        y1=i[1]-mean1
        k=x1*y1
        cnt1+=k
        cnt2+=x1**2
    b_1=cnt1/cnt2
    b_0=mean1-(b_1*mean0)
    reg=regression(b_0,b_1)
    tu=(b_0,b_1,reg)
    return tu

#Done 
def predict(year):
    p=compute_betas()
    y=p[0]+(p[1]*year)
    return y   

     
#Done
def iterate_normalized(T,eta):
    k=get_dataset()
    length=len(k)
    cnt=0
    for i in k:
        cnt+=i[0]
    mean0=cnt/length
    cnt1=0
    for j in k:
        p=(j[0]-mean0)**2
        p=round(p,2)
        cnt1+=p
    vari=cnt1/(length-1)
    covar=vari**0.5
    for a in k:
        p=(a[0]-mean0)/covar
        a[0]=p
    b_0=0
    b_1=0
    for i in range(1,T+1):
        x=grad_des(b_0,b_1,k)
        b0_new=b_0-eta*(x[0])
        b1_new=b_1-eta*(x[1])
        reg=regr(b0_new,b1_new,k)
        print(str(i)+" "+"{:.2f}".format(b0_new)+" "+"{:.2f}".format(b1_new)+" "+"{:.2f}".format(reg))
        b_0=b0_new
        b_1=b1_new

#Done
def sgd(T,eta):
    k=get_dataset()
    length=len(k)
    cnt=0
    for i in k:
        cnt+=i[0]
    mean0=cnt/length
    cnt1=0
    for j in k:
        p=(j[0]-mean0)**2
        cnt1+=p
    vari=cnt1/(length-1)
    covar=vari**0.5
    for a in k:
        p=(a[0]-mean0)/covar
        a[0]=p
    b_0=0
    b_1=0
    for i in range(1,T+1):
        x=grad_des1(b_0,b_1,k)
        b0_new=b_0-eta*(x[0])
        b1_new=b_1-eta*(x[1])
        reg=regr(b0_new,b1_new,k)
        print(str(i)+" "+"{:.2f}".format(b0_new)+" "+"{:.2f}".format(b1_new)+" "+"{:.2f}".format(reg))
        b_0=b0_new
        b_1=b1_new
    
    
# x=get_dataset()
# print(x)
# print_stats(x)
# k=regression(300,-0.1)
# print(k)      
# s=gradient_descent(0,0)
# print(s)
# iterate_gradient(5,1e-6)
# a=compute_betas()
# print(a)
# b=predict(2021)
# print(b)
# iterate_normalized(5, 0.01)
# sgd(5,0.1)