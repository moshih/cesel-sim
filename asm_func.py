# -*- coding: utf-8 -*-
"""
Created on Wed May 17 21:11:36 2017

@author: mshih
"""
from __future__ import division
from __future__ import print_function

import numpy as np

def get_index(row,index):
    if row==0 or row==1:
        return [4*index+3,4*index+2,4*index+1,4*index]
    else:
        return [4*index,4*index+1,4*index+2,4*index+3]

def sigma_one_asm_part1(p,data,result,calca,calcb,constant):
    p.permute(calca,data,constant)
    p.permute(data,data,constant)
    p.ror(data,constant,data)
    p.ror(calca,constant,calca)
    p.and_(data,data,constant)
    p.and_(calca,calca,constant)
    p.add8(data,calca,data)
    p.and_(result,data,data)


    p.permute(calca,data,constant)
    p.ror(data,constant,data)
    p.ror(calca,constant,calca)
    p.and_(data,data,constant)
    p.and_(calca,calca,constant)
    p.add8(data,calca,data)
    p.permute(calcb,data,constant)
    p.xor(result,result,calcb)
    
    p.permute(calca,data,constant)
    p.permute(data,data,constant)
    p.ror(data,constant,data)
    p.ror(calca,constant,calca)
    p.and_(data,data,constant)
    p.and_(calca,calca,constant)
    p.add8(data,calca,data)
    p.and_(data,data,constant)
    
    
    p.permute(calcb,data,constant)
    p.xor(result,result,calcb)
    
def sigma_one_asm_part2(i,data,result,calca,calcb,constant):
    i.regfile[result]=[ 27,26,25,24,  31,30,29,28, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,  0,0,0,0, 0,0,0,0]
    i.step()
    
    #print("the data is")
    #print ([np.binary_repr(n, width=8) for n in i.regfile[2][0:8]])
    
    i.regfile[constant]=[1,2,3,0, 5,6,7,4, 9,10,11,8, 13,14,15,12, 17,18,19,16, 21,22,23,20, 25,26,27,24, 29,30,31,28]
    i.step()
    i.regfile[constant]=[2,3,0,1, 6,7,4,5  ,10,11,8,9 ,14,15,12,13 ,18,19,16,17 ,22,23,20,21 ,26,27,24,25 ,30,31,28,29]
    i.step()
    i.regfile[constant]=[0b00000001 for x in range(0,32)]
    i.step()
    i.step()
    i.regfile[constant]=[0b01111111 for x in range(0,32)]
    i.step()
    i.regfile[constant]=[0b10000000 for x in range(0,32)]
    i.step()
    i.step()
    #print ([np.binary_repr(n, width=8) for n in i.regfile[data][0:4]])
    sumationone_one=np.copy(i.regfile[data][0:4])
    i.step()


    ##############################################################################
    i.regfile[constant]=[3,0,1,2,  7,4,5,6, 11,8,9,10, 15,12,13,14, 19,16,17,18, 23,20,21,22, 27,24,25,26, 31,28,29,30]

    i.step()
    i.regfile[constant]=[0b00000010 for x in range(0,32)]
    i.step()
    i.step()
    i.regfile[constant]=[0b00111111 for x in range(0,32)]
    i.step()
    i.regfile[constant]=[0b11000000 for x in range(0,32)]
    i.step()
    i.step()
    #print ([np.binary_repr(n, width=8) for n in i.regfile[data][0:4]])
    sumationone_two=np.copy(i.regfile[data][4:8])
    i.regfile[constant]=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    i.step()
    i.step()
    
        ##############################################################################

    i.regfile[constant]=[1,2,3,0, 5,6,7,4, 9,10,11,8, 13,14,15,12, 17,18,19,16, 21,22,23,20, 25,26,27,24, 29,30,31,28]
    i.step()
    i.regfile[constant]=[2,3,0,1, 6,7,4,5  ,10,11,8,9 ,14,15,12,13 ,18,19,16,17 ,22,23,20,21 ,26,27,24,25 ,30,31,28,29]
    i.step()
    i.regfile[constant]=[0b00000111 for x in range(0,32)]
    i.step()
    i.step()
    i.regfile[constant]=[0b00000001 for x in range(0,32)]
    i.step()
    i.regfile[constant]=[0b11111110 for x in range(0,32)]
    i.step()
    i.step()
    i.regfile[constant]=[0b11111111 for x in range(0,32)]
    i.regfile[constant][0]=0b00000000
    i.regfile[constant][1]=0b00111111
    i.regfile[constant][4]=0b00000000
    i.regfile[constant][5]=0b00111111
    i.regfile[constant][8]=0b00000000
    i.regfile[constant][9]=0b00111111
    i.regfile[constant][12]=0b00000000
    i.regfile[constant][13]=0b00111111
    i.regfile[constant][16]=0b00000000
    i.regfile[constant][17]=0b00111111
    i.regfile[constant][20]=0b00000000
    i.regfile[constant][21]=0b00111111
    i.regfile[constant][24]=0b00000000
    i.regfile[constant][25]=0b00111111
    i.regfile[constant][28]=0b00000000
    i.regfile[constant][29]=0b00111111
    i.step()
    
    
    #print ([np.binary_repr(n, width=8) for n in i.regfile[data][0:4]])
    sumationone_three=np.copy(i.regfile[data][8:12])
    i.regfile[constant]=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    i.step()
    i.step()
    
    
    
def sigma_zero_asm_part1(p,data,result,calca,calcb,calcc):
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
    p.permute(data,data,calcb)
    #set set calcb as 2
    p.ror(data,calcb,data)
    p.ror(calca,calcb,calca)
    #set set calcb as 0b00111111
    p.and_(data,data,calcb)
    #set set calcb as 0b11000000
    p.and_(calca,calca,calcb)
    p.add8(data,calca,data)
    p.permute(calcc,data,calcb)
    p.xor(result,result,calcc)
    
    
    p.permute(calca,data,calcb)
    #set set calcb as 2
    p.permute(data,data,calcb)
    #set set calcb as 2
    p.ror(data,calcb,data)
    p.ror(calca,calcb,calca)
    #set set calcb as 0b00111111
    p.and_(data,data,calcb)
    #set set calcb as 0b11000000
    p.and_(calca,calca,calcb)
    p.add8(data,calca,data)
    
    p.and_(data,data,calcb)
    
    p.permute(calcc,data,calcb)
    p.xor(result,result,calcc)
    
def sigma_zero_asm_part2(i,data,result,calca,calcb,calcc):
    i.regfile[calcb]=[3,0,1,2, 7,4,5,6, 11,8,9,10, 15,12,13,14, 19,16,17,18, 23,20,21,22, 27,24,25,26, 31,28,29,30]
    i.step()
    i.regfile[calcb]=[0b00000111 for x in range(0,32)]
    i.step()
    i.step()
    #i.regfile[calcb]=[0b00111111 for x in range(0,32)]
    i.regfile[calcb]=[0b00000001 for x in range(0,32)]
    i.step()
    #i.regfile[calcb]=[0b11000000 for x in range(0,32)]
    i.regfile[calcb]=[0b11111110 for x in range(0,32)]
    i.step()
    i.step()
    #print ([np.binary_repr(n, width=8) for n in i.regfile[data][0:4]])
    sumationzero_one=np.copy(i.regfile[data][0:4])
    i.step()


    ##############################################################################
    #4,5,6,7, 8,9,10,11, 12,13,14,15, 16,17,18,19, 20,21,22,23, ,24,25,26,27, 28,29,30,31
    #rotate by 28 to get the first part of sumationzero
    i.regfile[calcb]=[2,3,0,1, 6,7,4,5  ,10,11,8,9 ,14,15,12,13 ,18,19,16,17 ,22,23,20,21 ,26,27,24,25 ,30,31,28,29]
    i.step()
    i.regfile[calcb]=[3,0,1,2,  7,4,5,6, 11,8,9,10, 15,12,13,14, 19,16,17,18, 23,20,21,22, 27,24,25,26, 31,28,29,30]
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
    #print ([np.binary_repr(n, width=8) for n in i.regfile[data][0:4]])
    sumationzero_two=np.copy(i.regfile[data][4:8])
    i.regfile[calcb]=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    i.step()
    i.step()
    
    
        ##############################################################################
    #4,5,6,7, 8,9,10,11, 12,13,14,15, 16,17,18,19, 20,21,22,23, ,24,25,26,27, 28,29,30,31
    #rotate by 28 to get the first part of sumationzero
     #i.regfile[calcb]=[0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15, 16,17,18,19, 20,21,22,23, 24,25,26,27, 28,29,30,31]
    #print ([np.binary_repr(n, width=8) for n in i.regfile[data][0:4]])
    i.regfile[calcb]=[1,2,3,0, 5,6,7,4, 9,10,11,8, 13,14,15,12, 17,18,19,16, 21,22,23,20, 25,26,27,24, 29,30,31,28]
    i.step()
    #i.regfile[calcb]=[1,2,3,0, 5,6,7,4, 9,10,11,8, 13,14,15,12, 17,18,19,16, 21,22,23,20, 25,26,27,24, 29,30,31,28]
    i.regfile[calcb]=[2,3,0,1, 6,7,4,5  ,10,11,8,9 ,14,15,12,13 ,18,19,16,17 ,22,23,20,21 ,26,27,24,25 ,30,31,28,29]
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
    
    i.regfile[calcb]=[0b11111111 for x in range(0,32)]
    i.regfile[calcb][0]=0b00011111
    i.regfile[calcb][4]=0b00011111
    i.regfile[calcb][8]=0b00011111
    i.regfile[calcb][12]=0b00011111
    i.regfile[calcb][16]=0b00011111
    i.regfile[calcb][20]=0b00011111
    i.regfile[calcb][24]=0b00011111
    i.regfile[calcb][28]=0b00011111
    i.step()
    #print ([np.binary_repr(n, width=8) for n in i.regfile[data][0:4]])
    sumationzero_three=np.copy(i.regfile[data][8:12])
    i.regfile[calcb]=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    i.step()
    i.step()
    
