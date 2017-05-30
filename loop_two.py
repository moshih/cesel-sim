# -*- coding: utf-8 -*-
"""
Created on Tue May 23 23:19:08 2017

@author: mshih
"""

from __future__ import division
from __future__ import print_function

import numpy as np
from asm_func import sigma_one_asm_part1,sigma_one_asm_part2,get_index
from loop_two_functions import sumation_one_asm_part1, sumation_one_asm_part2,sumation_zero_asm_part1, sumation_zero_asm_part2,ch_part1,ch_part2
R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15 = np.arange(16)

##R8 is the data!!!!!!!!!!!!!!!!!
K=[[0x42,0x8a,0x2f,0x98], [0x71,0x37,0x44,0x91], [0xb5,0xc0,0xfb,0xcf], [0xe9,0xb5,0xdb,0xa5], [0x39,0x56,0xc2,0x5b], [0x59,0xf1,0x11,0xf1], [0x92,0x3f,0x82,0xa4], [0xab,0x1c,0x5e,0xd5],  
[0xd8,0x07,0xaa,0x98], [0x12,0x83,0x5b,0x01], [0x24,0x31,0x85,0xbe], [0x55,0x0c,0x7d,0xc3], [0x72,0xbe,0x5d,0x74], [0x80,0xde,0xb1,0xfe], [0x9b,0xdc,0x06,0xa7], [0xc1,0x9b,0xf1,0x74],   
[0xe4,0x9b,0x69,0xc1], [0xef,0xbe,0x47,0x86], [0x0f,0xc1,0x9d,0xc6], [0x24,0x0c,0xa1,0xcc], [0x2d,0xe9,0x2c,0x6f], [0x4a,0x74,0x84,0xaa], [0x5c,0xb0,0xa9,0xdc], [0x76,0xf9,0x88,0xda],   
[0x98,0x3e,0x51,0x52], [0xa8,0x31,0xc6,0x6d], [0xb0,0x03,0x27,0xc8], [0xbf,0x59,0x7f,0xc7], [0xc6,0xe0,0x0b,0xf3], [0xd5,0xa7,0x91,0x47], [0x06,0xca,0x63,0x51], [0x14,0x29,0x29,0x67],   
[0x27,0xb7,0x0a,0x85], [0x2e,0x1b,0x21,0x38], [0x4d,0x2c,0x6d,0xfc], [0x53,0x38,0x0d,0x13], [0x65,0x0a,0x73,0x54], [0x76,0x6a,0x0a,0xbb], [0x81,0xc2,0xc9,0x2e], [0x92,0x72,0x2c,0x85],   
[0xa2,0xbf,0xe8,0xa1], [0xa8,0x1a,0x66,0x4b], [0xc2,0x4b,0x8b,0x70], [0xc7,0x6c,0x51,0xa3], [0xd1,0x92,0xe8,0x19], [0xd6,0x99,0x06,0x24], [0xf4,0x0e,0x35,0x85], [0x10,0x6a,0xa0,0x70],   
[0x19,0xa4,0xc1,0x16], [0x1e,0x37,0x6c,0x08], [0x27,0x48,0x77,0x4c], [0x34,0xb0,0xbc,0xb5], [0x39,0x1c,0x0c,0xb3], [0x4e,0xd8,0xaa,0x4a], [0x5b,0x9c,0xca,0x4f], [0x68,0x2e,0x6f,0xf3],   
[0x74,0x8f,0x82,0xee], [0x78,0xa5,0x63,0x6f], [0x84,0xc8,0x78,0x14], [0x8c,0xc7,0x02,0x08], [0x90,0xbe,0xff,0xfa], [0xa4,0x50,0x6c,0xeb], [0xbe,0xf9,0xa3,0xf7], [0xc6,0x71,0x78,0xf2]]   
def tovalue(i):
    a=[]
    for x in range(0,8):
        a=a+[np.uint32(i[4*x])*256*256*256+np.uint32(i[4*x+1])*256*256+np.uint32(i[4*x+2])*256+np.uint32(i[4*x+3])]
    return a;
def tovalue1(i):
    a=[]
    for x in range(0,8):
        a=a+[np.uint32(i[4*x+3])*256*256*256+np.uint32(i[4*x+2])*256*256+np.uint32(i[4*x+1])*256+np.uint32(i[4*x+0])]
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


from loop_two_functions import sumation_one_asm_part1, sumation_one_asm_part2,sumation_zero_asm_part1, sumation_zero_asm_part2,maj_part1,maj_part2


def loop2_part1(p,iteration):
    #load constant into R9
    p.permute(R9,R8,R9)
    p.and_(R11,R8,R8)
    sumation_one_asm_part1(p,R11,R10,R12,R13,R14)
    #load constant into R9
    #load constant into R11
    p.permute(R11,R8,R11)
    #load constant into R12
    p.permute(R12,R8,R12)
    #load constant into R13
    p.permute(R13,R8,R13)
    ch_part1(p,R11,R12,R13,R10,R14)
    #i.regfile[R9]=add(i.regfile[R9],i.regfile[R10])
    #load constant into R10
    
    ##geting W
    row=int((iteration-iteration%8)/8)
    #load constant into R9
    #load constant into R10
    p.permute(R10,row,R10)
    #load constant into R9
    #load constant into R10
    p.permute(R9,R9,R10)
    #load constant into R10
    p.and_(R9,R9,R10)
    #load constant into R10
    p.permute(R11,R8,R10)
    sumation_zero_asm_part1(p,R11,R10,R12,R13,R14)
    #load constant into R11
    
    p.permute(R11,R8,R11)
    #load constant into R12
    p.permute(R12,R8,R12)
    #load constant into R13
    p.permute(R13,R8,R13)
    
    maj_part1(p,R11,R12,R13,R14,R15)
    #load constant into R10
    #load constant into R11
    #add
    p.and_(R10,R11,R10)
    #i.regfile[R9]=add(i.regfile[R9],i.regfile[R10])
    p.permute(R10,R8,R10)
    #load constant into R10
    p.and_(R10,R11,R10)
    #load constant into R9
    #i.regfile[R9]=add(i.regfile[R9],i.regfile[R10])
    #load constant into R11
    p.and_(R8,R9,R9)
    
def loop2_part2(i,iteration):
    i.regfile[R9]=[0 for x in range(0,16)]+get_index(8,7)+[0 for x in range(20,32)]
    i.step()
    i.step()
    sumation_one_asm_part2(i,R11,R10,R12,R13,R14)
    i.regfile[R9]=add(i.regfile[R9],i.regfile[R10])
    i.regfile[R11]=[0 for x in range(0,16)]+get_index(8,4)+[0 for x in range(20,32)]
    i.step()
    i.regfile[R12]=[0 for x in range(0,16)]+get_index(8,5)+[0 for x in range(20,32)]
    i.step()
    i.regfile[R13]=[0 for x in range(0,16)]+get_index(8,6)+[0 for x in range(20,32)]
    i.step()
    ch_part2(i,R11,R12,R13,R10,R14)
    i.regfile[R9]=add(i.regfile[R9],i.regfile[R10])
    i.regfile[R10]=[0 for x in range(0,16)]+K[iteration]+[0 for x in range(20,32)]
    

    row=int((iteration-iteration%8)/8)
    col=iteration%8
    i.regfile[R9]=add(i.regfile[R9],i.regfile[R10])
    i.regfile[R10]=[0 for x in range(0,16)]+get_index(row,col)+[0 for x in range(20,32)]
    i.step()

    i.regfile[R9]=add(i.regfile[R9],i.regfile[R10])
    i.regfile[R10]=get_index(8,4)+[0 for x in range(0,12)]+get_index(8,4)+[0 for x in range(16,28)]
    i.step()
    i.regfile[R10]=[255,255,255,255]+[0 for x in range(0,12)]+[255,255,255,255]+[0 for x in range(16,28)]
    i.step()
    
    i.regfile[R10]=[0,1,2,3]+[0 for x in range(0,28)]
    i.step()
    sumation_zero_asm_part2(i,R11,R10,R12,R13,R14)
    
    
    i.regfile[R11]=[0,1,2,3]+[0 for x in range(0,28)]
    i.step()
    i.regfile[R12]=[4,5,6,7]+[0 for x in range(0,28)]
    i.step()
    i.regfile[R13]=[8,9,10,11]+[0 for x in range(0,28)]
    i.step()
    
    maj_part2(i,R11,R12,R13,R14,R15)
    
    i.regfile[R10]=add(i.regfile[R10],i.regfile[R14])
    i.regfile[R11]=[255,255,255,255]+[0 for x in range(0,28)]
    i.step()
    i.regfile[R9]=add(i.regfile[R9],i.regfile[R10])
    
    i.regfile[R10]=[0,0,0,0, 0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15, 16,17,18,19, 20,21,22,23, 24,25,26,27]
    i.step()
    
    i.regfile[R11]=[0,0,0,0]+[255 for x in range(0,28)]
    i.step()
    i.regfile[R9]=add(i.regfile[R9],i.regfile[R10])
    i.step()
    
    
