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




def transfer_two_words_part1(p,rega,offseta,regb,offsetb,calca,calcb):
    #load constant into calca
    p.permute(calca,rega,calca)
    #load constant into calcb
    p.and_(calcb,calca,calcb)
    #load constant into calc
    p.and_(regb,calca,regb)
    p.add8(regb,regb,calcb )
    
def transfer_two_words_part2(i,rega,offseta,regb,offsetb,calca,calcb):
    
    i.regfile[calca]=[0 for x in range(0,32)]
    i.regfile[calca][offsetb:offsetb+8]=[x for x in range(offseta,offseta+8)]
    i.step()
    i.regfile[calcb]=[0 for x in range(0,32)]
    i.regfile[calcb][offsetb:offsetb+8]=[255 for x in range(offsetb,offsetb+8)]
    i.step()
    
    i.regfile[calca]=[255 for x in range(0,32)]
    i.regfile[calca][offsetb:32]=[0 for x in range(offsetb, 32)]
    i.step()
    i.step()
def loop1_batch_part1(p,result,calca,calcb,calcc,calcd,calce,winputa,winputb):
    #load constant into calca
    p.permute(calca,winputa[0],calca)
    sigma_one_asm_part1(p,calca,result,calcb,calcc,calcd)
    #load constant into calca
    p.permute(calca,winputa[2],calca)
    #load constant into calcb
    p.and_(calca,calca,calcb)
    #load constant into calcb
    p.permute(calcb,winputb[2],calcb)
    #load constant into calcc
    p.and_(calcb,calcb,calcc)
    p.add8(calca,calca,calcb)
    #load constant into calcb
    p.permute(calcb,winputa[6],calcb)
    #load constant into calcc
    p.and_(calcb,calcb,calcc)
    #load constant into calca
    p.permute(calca,winputa[4],calca)
    #load constant into calcb
    p.and_(calca,calca,calcb)
    #load constant into calcb
    p.permute(calcb,winputb[4],calcb)
    #load constant into calcc
    p.and_(calcb,calcb,calcc)
    p.add8(calca,calca,calcb)
    asm_func.sigma_zero_asm_part1(p, calca,calcb,calcc,calcd,calce)

def loop1_batch_part2(i,result,calca,calcb,calcc,calcd,calce,winputa,winputb,debug=0):
    i.regfile[calca]=get_index(winputa[0],winputa[1])+get_index(winputb[0],winputb[1])+[0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]
    i.step()

    if debug==1:
        print(" ")
        print("pre sigma 1")
        print(tovalue(i.regfile[calca]))
    sigma_one_asm_part2(i,calca,result,calcb,calcc,calcd)

    
    i.regfile[calca]=get_index(winputa[2],winputa[3])+[ 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]
    i.step()
    
    i.regfile[calcb]=[255,255,255,255, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0 ]
    i.step()
    
    i.regfile[calcb]=[ 0,0,0,0]+get_index(winputb[2],winputb[3])+[  0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]
    i.step()
    i.regfile[calcc]=[0,0,0,0,255,255,255,255,  0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0 ]
    i.step()
    i.step()
    
    
    i.regfile[calcb]=get_index(winputa[6],winputa[7])+get_index(winputb[6],winputb[7])+[  0,0,0,0,  0,0,0,0,  0,0,0,0,  0,0,0,0,  0,0,0,0, 0,0,0,0]
    i.step()
    i.regfile[calcc]=[255,255,255,255, 255,255,255,255, 0,0,0,0, 0,0,0,05, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0 ]
    i.step()

    if debug==1:
        print(tovalue(i.regfile[calca]))
        print(tovalue(i.regfile[calcb]))
    i.regfile[calca]=add(i.regfile[calca],i.regfile[calcb])

    if debug==1:
        print(tovalue(i.regfile[result]))
        
    i.regfile[result]=add(i.regfile[result],i.regfile[calca])
    
    
    i.regfile[calca]=get_index(winputa[4],winputa[5])+[ 0,0,0,0,0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]
    i.step()
    i.regfile[calcb]=[255,255,255,255, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0 , 0,0,0,0, 0,0,0,0, 0,0,0,0 ]
    i.step()
    
    i.regfile[calcb]=[ 0,0,0,0]+get_index(winputb[4],winputb[5])+[  0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]
    i.step()
    i.regfile[calcc]=[0,0,0,0,255,255,255,255,  0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0 ]
    i.step()
    i.step()
    

    asm_func.sigma_zero_asm_part2(i, calca,calcb,calcc,calcd,calce)

    if debug==1:
        print(tovalue(i.regfile[calcb]))
        print(" ")
    i.regfile[result]=add(i.regfile[result],i.regfile[calcb])
    if debug==1:
        print("finished of full batch 1")
        print(tovalue(i.regfile[result]))
 