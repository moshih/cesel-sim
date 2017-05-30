# -*- coding: utf-8 -*-
"""
Created on Thu May 18 17:14:10 2017

@author: mshih
"""

import numpy as np

R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15 = np.arange(16)
import asm_func
from asm_func import sigma_one_asm_part1,sigma_one_asm_part2,get_index

def tovalue(i):
    a=[]
    for x in range(0,8):
        a=a+[np.uint32(i[4*x])*256*256*256+np.uint32(i[4*x+1])*256*256+np.uint32(i[4*x+2])*256+np.uint32(i[4*x+3])]
    #b = np.uint32(i[4])*256*256*256+np.uint32(i[5])*256*256+np.uint32(i[6])*256+np.uint32(i[7])
    #print(a,b)
    return a;
def tovalue1(i):
    a=[]
    for x in range(0,8):
        a=a+[np.uint32(i[4*x+3])*256*256*256+np.uint32(i[4*x+2])*256*256+np.uint32(i[4*x+1])*256+np.uint32(i[4*x+0])]
    #b = np.uint32(i[4])*256*256*256+np.uint32(i[5])*256*256+np.uint32(i[6])*256+np.uint32(i[7])
    #print(a,b)
    return a;
def tobin(i):
    a=[]
    for x in range(0,8):
        a=a+[(i[x]&0b11111111000000000000000000000000)>>24 ,(i[x]&0b111111110000000000000000)>>16, (i[x]&0b1111111100000000)>>8,i[x]&0b11111111]
    return a;
def tobin1(i):
    a=[]
    for x in range(0,8):
        a=a+[i[x]&0b11111111,(i[x]&0b1111111100000000)>>8,(i[x]&0b111111110000000000000000)>>16, (i[x]&0b11111111000000000000000000000000)>>24 ]
    return a;
def add (a,b):
    c=tovalue(a)
    d=tovalue(b)
    for x in range(0,8):
        c[x]=c[x]+d[x]
    return tobin(c)


#2 0 | 1 3 | 0 3 | 0 2
#2 1 | 1 4 | 0 4 | 0 3
#winputa=[2,0,1,3,0,3,0,2]
#winputb=[2,1,1,4,0,4,0,3]
#winputa=[1, 6 , 1 ,1 , 0, 1 , 0, 0]
#winputb=[1, 7 ,1 ,2 , 0, 2 , 0, 1]
winputa=[2 ,2,  1, 5,  0, 5,  0, 4]
winputb=[2 ,3,  1, 6,  0, 6,  0, 5]
result=R3
calca=R4
calcb=R5
calcc=R6
calcd=R7
calce=R8
def loop1_batch_part1(p):
    p.permute(calca,winputa[0],calca)
    sigma_one_asm_part1(p,calca,result,calcb,calcc,calcd)
    
    
    p.permute(calca,winputa[2],calca)
    p.and_(calca,calca,calcb)
    
    #p.permute(R5,R2,R5)
    #p.and_(R5,R5,R6)
    #p.add8(R4,R4,R5)
    
    p.permute(calcb,winputa[6],calcb)
    p.and_(calcb,calcb,calcc)
    
    #p.permute(R6,R1,R6)
    #p.and_(R6,R6,R7)
    #p.add8(R5,R5,R6)
    
    ##add_32 bit word R4, R4,R5
    ##add_32 bit word R3,R3,R4
    
    #sigma_zero
    p.permute(calca,winputa[4],calca)
    p.and_(calca,calca,calcb)
    
    #p.permute(R5,R1,R5)
    #p.and_(R5,R5,R6)
    #p.add8(R4,R4,R5)
    asm_func.sigma_zero_asm_part1(p, calca,calcb,calcc,calcd,calce)
    ##p.add8(R3,R3,R5)
    
    
def loop1_batch_part2(i):
    i.regfile[calca]=get_index(winputa[0],winputa[1])+get_index(winputb[0],winputb[1])+[0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]
    i.step()
    #print(tovalue(i.regfile[4]))
    print("pre sigma 1")
    print(tovalue(i.regfile[calca]))
    sigma_one_asm_part2(i,calca,result,calcb,calcc,calcd)
    
    #print("sigma1 of full batch 1")
    #print(tovalue(i.regfile[3]))
    
    ### At this point, R3 holds sigma 1 for this batch (first full batch)
    
    i.regfile[calca]=get_index(winputa[2],winputa[3])+get_index(winputb[2],winputb[3])+[ 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]
    i.step()
    i.regfile[calcb]=[255,255,255,255, 255,255,255,255, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0 ]
    i.step()
    #print(tovalue(i.regfile[4]))
    #i.regfile[5]=[ 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]+get_index(2,0)+get_index(2,1)+get_index(2,2)
    #i.step()
    #i.regfile[6]=[0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,  255,255,255,255, 255,255,255,255, 255,255,255,255 ]
    #i.step()
    #print(tovalue(i.regfile[5]))
    #i.step()
    #print(tovalue(i.regfile[4]))
    
    i.regfile[calcb]=get_index(winputa[6],winputa[7])+get_index(winputb[6],winputb[7])+[  0,0,0,0,  0,0,0,0,  0,0,0,0,  0,0,0,0,  0,0,0,0, 0,0,0,0]
    i.step()
    i.regfile[calcc]=[255,255,255,255, 255,255,255,255, 0,0,0,0, 0,0,0,05, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0 ]
    i.step()
    #print(tovalue(i.regfile[4]))
    #print(tovalue(i.regfile[5]))
    
    #i.regfile[6]=[ 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]+get_index(1,0)+get_index(1,1)
    #i.step()
    #i.regfile[7]=[0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,  0,0,0,0, 255,255,255,255, 255,255,255,255 ]
    #i.step()
    #print(tovalue(i.regfile[6]))
    #print(tovalue(i.regfile[5]))
    #i.step()
    #print(tovalue(i.regfile[5]))
    ##i.step()
    print(tovalue(i.regfile[calca]))
    print(tovalue(i.regfile[calcb]))
    i.regfile[calca]=add(i.regfile[calca],i.regfile[calcb])
    #print("some of two of full batch 1")
    #print(tovalue(i.regfile[4]))
    print(tovalue(i.regfile[result]))
    i.regfile[result]=add(i.regfile[result],i.regfile[calca])
    
    
    i.regfile[calca]=get_index(winputa[4],winputa[5])+get_index(winputb[4],winputb[5])+[ 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]
    i.step()
    i.regfile[calcb]=[255,255,255,255, 255,255,255,255, 0,0,0,0, 0,0,0,0, 0,0,0,0 , 0,0,0,0, 0,0,0,0, 0,0,0,0 ]
    i.step()
    #print(tovalue(i.regfile[4]))
    #i.regfile[5]=[ 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]+get_index(1,0)+get_index(1,1)+get_index(1,2)
    #i.step()
    #i.regfile[6]=[0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,  255,255,255,255, 255,255,255,255, 255,255,255,255 ]
    #i.step()
    #print(tovalue(i.regfile[5]))
    #i.step()
    #print(tovalue(i.regfile[4]))
    asm_func.sigma_zero_asm_part2(i, calca,calcb,calcc,calcd,calce)
    #print(tovalue(i.regfile[5]))
    #print(tovalue(i.regfile[3]))
    print(tovalue(i.regfile[calcb]))
    i.regfile[result]=add(i.regfile[result],i.regfile[calcb])
    print("finished of full batch 1")
    print(tovalue(i.regfile[result]))
    print(i.regfile[result])
    #print ([np.binary_repr(n, width=8) for n in i.regfile[result][0:8]])