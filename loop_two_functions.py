# -*- coding: utf-8 -*-
"""
Created on Tue May 23 21:03:58 2017

@author: mshih
"""

from __future__ import division
from __future__ import print_function

import numpy as np

def ch_part1(p,x,y,z,result,calca):
    p.and_(result,x,y)
    ##have calca equal 255's
    p.xor(calca,calca,x)
    p.and_(calca,calca,z)
    p.xor(result,result,calca)
    
def ch_part2(i,x,y,z,result,calca):
    i.step()
    i.regfile[calca]=[255 for j in range(0, 32)]
    i.step()
    i.step()
    i.step()
    
def maj_part1(p,x,y,z,result,calca):
    p.and_(result,x,y)
    p.and_(calca,x,z)
    p.xor(result,result,calca)
    p.and_(calca,y,z)
    p.xor(result,result,calca)
    
def maj_part2(i,x,y,z,result,calca):
    i.step()
    i.step()
    i.step()
    i.step()
    i.step()
def tovalue(i):
    a=[]
    for x in range(0,8):
        a=a+[np.uint32(i[4*x])*256*256*256+np.uint32(i[4*x+1])*256*256+np.uint32(i[4*x+2])*256+np.uint32(i[4*x+3])]
    return a;

def sumation_one_asm_part1(p,data,result,calca,calcb,calcc):
    #load constant into calcb
    p.permute(calca,data,calcb)
    #load constant into calcb
    p.ror(data,calcb,data)
    p.ror(calca,calcb,calca)
    #load constant into calcb
    p.and_(data,data,calcb)
    #load constant into calcb
    p.and_(calca,calca,calcb)
    p.add8(data,calca,data)
    p.and_(result,data,data)

    #load constant into calcb
    p.permute(calca,data,calcb)
    #load constant into calcb
    p.ror(data,calcb,data)
    p.ror(calca,calcb,calca)
    #load constant into calcb
    p.and_(data,data,calcb)
    #load constant into calcb
    p.and_(calca,calca,calcb)
    p.add8(data,calca,data)
    p.xor(result,data,result)
    
    #load constant into calcb
    p.permute(calca,data,calcb)
    #load constant into calcb
    p.permute(data,data,calcb)
    #load constant into calcb
    p.ror(data,calcb,data)
    p.ror(calca,calcb,calca)
    #load constant into calcb
    p.and_(data,data,calcb)
    #load constant into calcb
    p.and_(calca,calca,calcb)
    p.add8(data,calca,data)
    p.xor(result,data,result)


    
def sumation_one_asm_part2(i,data,result,calca,calcb,calcc):
    i.regfile[calcb]=[3,0,1,2, 7,4,5,6, 11,8,9,10, 15,12,13,14, 19,16,17,18, 23,20,21,22, 27,24,25,26, 31,28,29,30]
    i.step()
    i.regfile[calcb]=[0b00000110 for x in range(0,32)]
    i.step()
    i.step()
    i.regfile[calcb]=[0b00000011 for x in range(0,32)]
    i.step()
    i.regfile[calcb]=[0b11111100 for x in range(0,32)]
    i.step()
    i.step()  
    i.step()

    i.regfile[calcb]=[3,0,1,2, 7,4,5,6, 11,8,9,10, 15,12,13,14, 19,16,17,18, 23,20,21,22, 27,24,25,26, 31,28,29,30]
    i.step()
    i.regfile[calcb]=[0b00000101 for x in range(0,32)]
    i.step()
    i.step()
    i.regfile[calcb]=[0b00000111 for x in range(0,32)]
    i.step()
    i.regfile[calcb]=[0b11111000 for x in range(0,32)]
    i.step()
    i.step()
    i.step()

    
    
    i.regfile[calcb]=[2,3,0,1, 6,7,4,5  ,10,11,8,9 ,14,15,12,13 ,18,19,16,17 ,22,23,20,21 ,26,27,24,25 ,30,31,28,29]
    i.step()
    i.regfile[calcb]=[3,0,1,2, 7,4,5,6, 11,8,9,10, 15,12,13,14, 19,16,17,18, 23,20,21,22, 27,24,25,26, 31,28,29,30]
    i.step()
    i.regfile[calcb]=[0b00000110 for x in range(0,32)]
    i.step()
    i.step()
    i.regfile[calcb]=[0b00000011 for x in range(0,32)]
    i.step()
    i.regfile[calcb]=[0b11111100 for x in range(0,32)]
    i.step()
    i.step()
    i.step()

    
###############################################################################################################################

def sumation_zero_asm_part1(p,data,result,calca,calcb,calcc):
    #load constant into calcb
    p.permute(calca,data,calcb)
    #load constant into calcb
    p.ror(data,calcb,data)
    p.ror(calca,calcb,calca)
    #load constant into calcb
    p.and_(data,data,calcb)
    #load constant into calcb
    p.and_(calca,calca,calcb)
    p.add8(data,calca,data)
    p.and_(result,data,data)

    #load constant into calcb
    p.permute(calca,data,calcb)
    #load constant into calcb
    p.permute(data,data,calcb)
    #load constant into calcb
    p.ror(data,calcb,data)
    p.ror(calca,calcb,calca)
    #load constant into calcb
    p.and_(data,data,calcb)
    #load constant into calcb#load constant into calcb
    p.and_(calca,calca,calcb)
    p.add8(data,calca,data)
    p.xor(result,data,result)
    
    #load constant into calcb
    p.permute(calca,data,calcb)
    #load constant into calcb
    p.permute(data,data,calcb)
    #load data into calcb
    p.ror(data,calcb,data)
    p.ror(calca,calcb,calca)
    #load constant into calcb
    p.and_(data,data,calcb)
    #load constant into calcb
    p.and_(calca,calca,calcb)
    p.add8(data,calca,data)
    p.xor(result,data,result)


    
def sumation_zero_asm_part2(i,data,result,calca,calcb,calcc):
    i.regfile[calcb]=[3,0,1,2, 7,4,5,6, 11,8,9,10, 15,12,13,14, 19,16,17,18, 23,20,21,22, 27,24,25,26, 31,28,29,30]
    i.step()
    i.regfile[calcb]=[0b00000010 for x in range(0,32)]
    i.step()
    i.step()
    i.regfile[calcb]=[0b00111111 for x in range(0,32)]
    i.step()
    i.regfile[calcb]=[0b11000000 for x in range(0,32)]
    i.step()
    i.step()
    i.step()

    i.regfile[calcb]=[2,3,0,1, 6,7,4,5  ,10,11,8,9 ,14,15,12,13 ,18,19,16,17 ,22,23,20,21 ,26,27,24,25 ,30,31,28,29]
    i.step()
    i.regfile[calcb]=[3,0,1,2, 7,4,5,6, 11,8,9,10, 15,12,13,14, 19,16,17,18, 23,20,21,22, 27,24,25,26, 31,28,29,30]
    i.step()
    i.regfile[calcb]=[0b00000011 for x in range(0,32)]
    i.step()
    i.step()
    i.regfile[calcb]=[0b00011111 for x in range(0,32)]
    i.step()
    i.regfile[calcb]=[0b11100000 for x in range(0,32)]
    i.step()
    i.step()
    i.step()
    
    i.regfile[calcb]=[2,3,0,1, 6,7,4,5  ,10,11,8,9 ,14,15,12,13 ,18,19,16,17 ,22,23,20,21 ,26,27,24,25 ,30,31,28,29]
    i.step()
    i.regfile[calcb]=[3,0,1,2, 7,4,5,6, 11,8,9,10, 15,12,13,14, 19,16,17,18, 23,20,21,22, 27,24,25,26, 31,28,29,30]
    i.step()
    i.regfile[calcb]=[0b00000001 for x in range(0,32)]
    i.step()
    i.step()
    i.regfile[calcb]=[0b01111111 for x in range(0,32)]
    i.step()
    i.regfile[calcb]=[0b10000000 for x in range(0,32)]
    i.step()
    i.step()
    i.step()

