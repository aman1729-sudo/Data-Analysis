import numpy as np
import matplotlib.pyplot
from numpy import random
import pandas as pd
from numpy import linalg as la
from numpy import genfromtxt
arr_m = np.array([[1,2,3],[4,5,6],[7,8,9]])
#creating arrays 
# =============================================================================
# arr_1 = [123,123,12,31,23321,42,4,213]
# np_arr_1 = np.array(arr_1)
# print(np_arr_1)
# np_arr2 = np.array([1,2,3,4,5],dtype = np.int8)
# print(np_arr2)
# np_arr3 = np.array([[123,123,24],[1,2,3]])
# print(np_arr3)
# np_zero = np.zeros(6)
# np_zero2 = np.ones((2,4))
# print(np_zero)
# print(np_arr_1.size)
# 
# =============================================================================
#random array using numpy
# =============================================================================
# arr_1 = np.random.randint(10,50,10)
# print(arr_1)
# arr_2 = np.random.randint(10,50,size = (2,5))
# print(arr_2)
# =============================================================================
#slicing and indexing 
# =============================================================================
# arr_m = np.array([[1,2,3],[4,5,6],[7,8,9]])
# arr_m[0,0] = 100
# arr_m.itemset((0,1),500)
# print(arr_m.shape)
# print(np.take(arr_m , [0,3]))
# np.put(arr_m,[2,4,7],[999,999,999])
# print(arr_m)
# print(arr_m[::-1])
# print(arr_m)
# even = arr_m[arr_m % 2 == 0]
# print(even)
# oper = arr_m[(arr_m % 2 == 0) & (arr_m > 6)]
# print(oper)
# =============================================================================
#reshaping arrays
# =============================================================================
# arr_2 = arr_m.reshape((1,9))
# print(arr_2)
# print(arr_m.transpose())
# arr_m.sort(axis = 1)
# print(arr_m.flatten())
# =============================================================================
#stacking and splitting
# =============================================================================
# arr1 = np.random.randint(10,size= (2,2))
# arr2 = np.random.randint(10,size= (2,2))
# arr3 = np.hstack((arr1,arr2))
# arr4 = np.vstack((arr1,arr2))
# arr5 = np.delete(arr1,1,0)
# arr6 = np.delete(arr2,1,0)
# print(np.column_stack((arr5,arr6)))
# ar7 = np.random.randint(10,size=(2,10))
# print(np.hsplit(ar7,5))
# print(np.vsplit(ar7,(2,4)))
# =============================================================================
#copying
# =============================================================================
# 1 and 2 gives shallow or copy that depends upon arr_m
# arr_c1 = arr_m
# arr_m[0,0] = 1000
# 
# arr_c2 = arr_m.view()
# arr_m[0,0] = 999
# .copy() gives true copy that don't depends upon arr_m or parent array
# arr_c3 = arr_m.copy()
# arr_m[0,0] = 888
# 
# =============================================================================
#mathematics with numpy
# =============================================================================
# arr2 = np.random.randint(10,size=(4))
# arr1 = np.random.randint(10,size=(4))
# print(arr1 / arr2 , " ", arr1 + arr2 , " " , arr1 * arr2)
# #get many random values under 1
# print(np.random.rand(6))
# #get any choice or random value from array
# print(random.choice(arr1))
# print(arr2.sum()) # sum of array
# print(arr2.cumsum()) #cumulative sum
# print(arr1.max(axis = 0)) #maximum along axis = 0
# print(arr1.min(axis = 0)) #minimum along axis = 0
# print(np.add(arr1,5)) #add 5 to every element of array
# arr3 = [[1,2],[3,4]]
# arr4 = [[11,12],[13,41]]
# print(np.remainder(arr4,arr3)) # remainder of array
# print(np.power(arr3,arr4)) # power of array slly get np.exp()
# print(np.sqrt(arr3),arr4) # squareroot slly get np.cbrt
# print(np.log(arr3)) #log of array slly get np.log10 or any base
# print(np.gcd.reduce([9,12,15]))
# print(np.lcm.reduce([9,12,15]))
# arr7 = np.random.randint(10,size = (5,3))
# mc = arr7.argmax(axis = 0) #get index of max values along any axis 
# print(arr7[mc])
# =============================================================================
#statistics
# =============================================================================
# arr1 = np.arange(10)
# print(arr1.mean()) # mean
# print(np.average(arr1)) #average of array
# print(np.var(arr1)) #variance 
# print(np.std(arr1)) #standard deviation
# #use np.nanmean , np.nanmedian and son on if any of the data values are not numbers 
# 
# =============================================================================
#Trignometry function
# =============================================================================
# a = np.linspace(-np.pi , np.pi , 200)
# print(matplotlib.pyplot.plot(a,np.sin(a)))# sin graph
# print(matplotlib.pyplot.plot(a,np.tan(a)))#tan graph
# print(np.sin(np.pi/2)) # sin values 
# print(np.arcsin(1)) # inverse of sin
# print(np.rad2deg(np.pi/2)) # radian to degree
# print(np.deg2rad(90)) # degree to radian 
# print(np.hypot(10,10)) # hypot. of right angled triangle
# =============================================================================
#Linear algebra with numpy
# =============================================================================
# arr_1 = np.random.randint(10,size= (3,3))
# arr_2 = np.random.randint(10,size = (3,3))
# arr_3 = np.random.randint(10,size = (3,3))
# print(np.dot(arr_1,arr_2)) #simple dot product
# print(la.multi_dot([arr_1,arr_2,arr_3])) #multiple array dot product using linalg
# print(np.inner(arr_1,arr_2)) #inner product
# arr_a = np.array([[[1,2],[3,4]],[[4,5],[6,7]]])
# arr_b = np.array([[1,2],[3,4]],dtype = object)
# print(np.tensordot(arr_a, arr_b)) #tensor product
# #Einstien Summation
# arr_aa = np.array([0,1])
# arr_bb = np.array([[1,2,3,4],[5,6,7,8]])
# print(np.einsum('i,ij -> i',arr_aa,arr_bb ))
# print(np.einsum('i->',[1,2,3])) #sum of array 
# print(la.matrix_power(arr_1, 2)) #rasie matrix to any power
# print(np.kron(arr_1, arr_b)) #kronecker product of arrays 
# print(la.eig(arr_1)) # eigen vector of arr_1
# print(la.eigvals(arr_1)) #eigen value of arr_1
# print(la.norm(arr_1)) #norm of arr_1
# print(la.inv(arr_1)) #inverse of arr_!
# print(la.det(arr_1)) #determinant fo arr_1
# 
# 
# =============================================================================
#system of linear equations
# =============================================================================
# #x + 4y = 10 , 6x + 18y = 42
# arr_1 = np.array([[1,4],[6,18]])
# arr_2 = np.array([10,42])
# print(la.solve(arr_1, arr_2))
# =============================================================================
#saving and loading numpy arrays 
# =============================================================================
# arr_1 = np.array([[1,2],[3,4]])
# np.save('andarray',arr_1)
# arr_2 = np.load('andarray.npy')
# print(arr_2)
# =============================================================================
