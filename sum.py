# -*- coding: utf-8 -*-
"""
Created on Tue May 23 21:03:58 2017

@author: mshih
"""

from __future__ import division
from __future__ import print_function

import numpy as np

def tovalue(i):
    a=[]
    for x in range(0,8):
        a=a+[np.uint32(i[4*x])*256*256*256+np.uint32(i[4*x+1])*256*256+np.uint32(i[4*x+2])*256+np.uint32(i[4*x+3])]
    #b = np.uint32(i[4])*256*256*256+np.uint32(i[5])*256*256+np.uint32(i[6])*256+np.uint32(i[7])
    #print(a,b)
    return a;

def sumation_one_asm_part1(p,data,result,calca,calcb,calcc):
    p.permute(calca,data,calcb)
    #set set calcb as 2
    p.ror(data,calcb,data)
    p.ror(calca,calcb,calca)
    #set set calcb as 0b00111111
    p.and_(data,data,calcb)
    #set set calcb as 0b11000000
    p.and_(calca,calca,calcb)
    p.add8(data,calca,data)
    p.and_(result,data,data)


    p.permute(calca,data,calcb)
    #set set calcb as 2
    p.ror(data,calcb,data)
    p.ror(calca,calcb,calca)
    #set set calcb as 0b00111111
    p.and_(data,data,calcb)
    #set set calcb as 0b11000000
    p.and_(calca,calca,calcb)
    p.add8(data,calca,data)
    p.xor(result,data,result)
    
    p.permute(calca,data,calcb)
    p.permute(data,data,calcb)
    #set set calcb as 2
    p.ror(data,calcb,data)
    p.ror(calca,calcb,calca)
    #set set calcb as 0b00111111
    p.and_(data,data,calcb)
    #set set calcb as 0b11000000
    p.and_(calca,calca,calcb)
    p.add8(data,calca,data)
    p.xor(result,data,result)


    
def sumation_one_asm_part2(i,data,result,calca,calcb,calcc):
    print("sumation one")
    print(tovalue(i.regfile[data]))
    #print ([np.binary_repr(n, width=8) for n in i.regfile[data][0:4]])
    i.regfile[calcb]=[3,0,1,2, 7,4,5,6, 11,8,9,10, 15,12,13,14, 19,16,17,18, 23,20,21,22, 27,24,25,26, 31,28,29,30]
    i.step()
    i.regfile[calcb]=[0b00000110 for x in range(0,32)]
    i.step()
    i.step()
    #i.regfile[calcb]=[0b00111111 for x in range(0,32)]
    i.regfile[calcb]=[0b00000011 for x in range(0,32)]
    i.step()
    #i.regfile[calcb]=[0b11000000 for x in range(0,32)]
    i.regfile[calcb]=[0b11111100 for x in range(0,32)]
    i.step()
    i.step()
    
    i.step()
    #print ([np.binary_repr(n, width=8) for n in i.regfile[result][0:4]])
    print(tovalue(i.regfile[result]))


    #i.regfile[constant]=[1,2,3,0, 5,6,7,4, 9,10,11,8, 13,14,15,12, 17,18,19,16, 21,22,23,20, 25,26,27,24, 29,30,31,28]
    i.regfile[calcb]=[3,0,1,2, 7,4,5,6, 11,8,9,10, 15,12,13,14, 19,16,17,18, 23,20,21,22, 27,24,25,26, 31,28,29,30]
    i.step()
    i.regfile[calcb]=[0b00000101 for x in range(0,32)]
    i.step()
    i.step()
    #i.regfile[calcb]=[0b00111111 for x in range(0,32)]
    i.regfile[calcb]=[0b00000111 for x in range(0,32)]
    i.step()
    #i.regfile[calcb]=[0b11000000 for x in range(0,32)]
    i.regfile[calcb]=[0b11111000 for x in range(0,32)]
    i.step()
    i.step()
    
    i.step()
    #print ([np.binary_repr(n, width=8) for n in i.regfile[data][0:4]])
    print(tovalue(i.regfile[data]))
    
    
    i.regfile[calcb]=[2,3,0,1, 6,7,4,5  ,10,11,8,9 ,14,15,12,13 ,18,19,16,17 ,22,23,20,21 ,26,27,24,25 ,30,31,28,29]
    i.step()
    i.regfile[calcb]=[3,0,1,2, 7,4,5,6, 11,8,9,10, 15,12,13,14, 19,16,17,18, 23,20,21,22, 27,24,25,26, 31,28,29,30]
    i.step()
    i.regfile[calcb]=[0b00000110 for x in range(0,32)]
    i.step()
    i.step()
    #i.regfile[calcb]=[0b00111111 for x in range(0,32)]
    i.regfile[calcb]=[0b00000011 for x in range(0,32)]
    i.step()
    #i.regfile[calcb]=[0b11000000 for x in range(0,32)]
    i.regfile[calcb]=[0b11111100 for x in range(0,32)]
    i.step()
    i.step()
    
    i.step()
    #print ([np.binary_repr(n, width=8) for n in i.regfile[data][0:4]])
    #print ([np.binary_repr(n, width=8) for n in i.regfile[result][0:4]])
    print(tovalue(i.regfile[data]))
    print("result")
    print(tovalue(i.regfile[result]))
    
###############################################################################################################################

def sumation_zero_asm_part1(p,data,result,calca,calcb,calcc):
    p.permute(calca,data,calcb)
    #set set calcb as 2
    p.ror(data,calcb,data)
    p.ror(calca,calcb,calca)
    #set set calcb as 0b00111111
    p.and_(data,data,calcb)
    #set set calcb as 0b11000000
    p.and_(calca,calca,calcb)
    p.add8(data,calca,data)
    p.and_(result,data,data)


    p.permute(calca,data,calcb)
    p.permute(data,data,calcb)
    #set set calcb as 2
    p.ror(data,calcb,data)
    p.ror(calca,calcb,calca)
    #set set calcb as 0b00111111
    p.and_(data,data,calcb)
    #set set calcb as 0b11000000
    p.and_(calca,calca,calcb)
    p.add8(data,calca,data)
    p.xor(result,data,result)
    
    p.permute(calca,data,calcb)
    p.permute(data,data,calcb)
    #set set calcb as 2
    p.ror(data,calcb,data)
    p.ror(calca,calcb,calca)
    #set set calcb as 0b00111111
    p.and_(data,data,calcb)
    #set set calcb as 0b11000000
    p.and_(calca,calca,calcb)
    p.add8(data,calca,data)
    p.xor(result,data,result)


    
def sumation_zero_asm_part2(i,data,result,calca,calcb,calcc):
    print("sumation one")
    print(tovalue(i.regfile[data]))
    #print ([np.binary_repr(n, width=8) for n in i.regfile[data][0:4]])
    i.regfile[calcb]=[3,0,1,2, 7,4,5,6, 11,8,9,10, 15,12,13,14, 19,16,17,18, 23,20,21,22, 27,24,25,26, 31,28,29,30]
    i.step()
    i.regfile[calcb]=[0b00000010 for x in range(0,32)]
    i.step()
    i.step()
    #i.regfile[calcb]=[0b00111111 for x in range(0,32)]
    i.regfile[calcb]=[0b00111111 for x in range(0,32)]
    i.step()
    #i.regfile[calcb]=[0b11000000 for x in range(0,32)]
    i.regfile[calcb]=[0b11000000 for x in range(0,32)]
    i.step()
    i.step()
    
    i.step()
    #print ([np.binary_repr(n, width=8) for n in i.regfile[result][0:4]])
    print(tovalue(i.regfile[result]))


    #i.regfile[constant]=[1,2,3,0, 5,6,7,4, 9,10,11,8, 13,14,15,12, 17,18,19,16, 21,22,23,20, 25,26,27,24, 29,30,31,28]
    i.regfile[calcb]=[2,3,0,1, 6,7,4,5  ,10,11,8,9 ,14,15,12,13 ,18,19,16,17 ,22,23,20,21 ,26,27,24,25 ,30,31,28,29]
    i.step()
    i.regfile[calcb]=[3,0,1,2, 7,4,5,6, 11,8,9,10, 15,12,13,14, 19,16,17,18, 23,20,21,22, 27,24,25,26, 31,28,29,30]
    i.step()
    i.regfile[calcb]=[0b00000011 for x in range(0,32)]
    i.step()
    i.step()
    #i.regfile[calcb]=[0b00111111 for x in range(0,32)]
    i.regfile[calcb]=[0b00011111 for x in range(0,32)]
    i.step()
    #i.regfile[calcb]=[0b11000000 for x in range(0,32)]
    i.regfile[calcb]=[0b11100000 for x in range(0,32)]
    i.step()
    i.step()
    
    i.step()
    #print ([np.binary_repr(n, width=8) for n in i.regfile[data][0:4]])
    print(tovalue(i.regfile[data]))
    
    
    i.regfile[calcb]=[2,3,0,1, 6,7,4,5  ,10,11,8,9 ,14,15,12,13 ,18,19,16,17 ,22,23,20,21 ,26,27,24,25 ,30,31,28,29]
    i.step()
    i.regfile[calcb]=[3,0,1,2, 7,4,5,6, 11,8,9,10, 15,12,13,14, 19,16,17,18, 23,20,21,22, 27,24,25,26, 31,28,29,30]
    i.step()
    i.regfile[calcb]=[0b00000001 for x in range(0,32)]
    i.step()
    i.step()
    #i.regfile[calcb]=[0b00111111 for x in range(0,32)]
    i.regfile[calcb]=[0b01111111 for x in range(0,32)]
    i.step()
    #i.regfile[calcb]=[0b11000000 for x in range(0,32)]
    i.regfile[calcb]=[0b10000000 for x in range(0,32)]
    i.step()
    i.step()
    
    i.step()
    #print ([np.binary_repr(n, width=8) for n in i.regfile[data][0:4]])
    #print ([np.binary_repr(n, width=8) for n in i.regfile[result][0:4]])
    print(tovalue(i.regfile[data]))
    print("result")
    print(tovalue(i.regfile[result]))
 
