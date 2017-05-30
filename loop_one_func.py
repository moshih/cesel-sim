# -*- coding: utf-8 -*-
"""
Created on Wed May 17 21:11:36 2017

@author: mshih
"""
from __future__ import division
from __future__ import print_function

import numpy as np
from shaloop import loop1
def get_index(row,index):
    if row==0 or row==1:
        return [4*index+3,4*index+2,4*index+1,4*index]
    else:
        return [4*index,4*index+1,4*index+2,4*index+3]

def sigma_one_asm_part1(p,data,result,calca,calcb,calcc):
    #load constant into calcc
    p.permute(calca,data,calcc)
    #load constant into calcc
    p.permute(data,data,calcc)
    #load constant into calcc
    p.ror(data,calcc,data)
    p.ror(calca,calcc,calca)
    #load constant into calcc
    p.and_(data,data,calcc)
    #load constant into calcc
    p.and_(calca,calca,calcc)
    p.add8(data,calca,data)
    p.and_(result,data,data)

    #load constant into calcc
    p.permute(calca,data,calcc)
    #load constant into calcc
    p.ror(data,calcc,data)
    p.ror(calca,calcc,calca)
    #load constant into calcc
    p.and_(data,data,calcc)
    #load constant into calcc
    p.and_(calca,calca,calcc)
    p.add8(data,calca,data)
    #load constant into calcc
    p.permute(calcb,data,calcc)
    p.xor(result,result,calcb)
    
    #load constant into calcc
    p.permute(calca,data,calcc)
    #load constant into calcc
    p.permute(data,data,calcc)
    #load constant into calcc
    p.ror(data,calcc,data)
    p.ror(calca,calcc,calca)
    #load constant into calcc
    p.and_(data,data,calcc)
    #load constant into calcc
    p.and_(calca,calca,calcc)
    p.add8(data,calca,data)
    #load constant into calcc
    p.and_(data,data,calcc)
    #load constant into calcc
    p.permute(calcb,data,calcc)
    p.xor(result,result,calcb)
    
def sigma_one_asm_part2(i,data,result,calca,calcb,calcc):    
    i.regfile[calcc]=[1,2,3,0, 5,6,7,4, 9,10,11,8, 13,14,15,12, 17,18,19,16, 21,22,23,20, 25,26,27,24, 29,30,31,28]
    i.step()
    i.regfile[calcc]=[2,3,0,1, 6,7,4,5  ,10,11,8,9 ,14,15,12,13 ,18,19,16,17 ,22,23,20,21 ,26,27,24,25 ,30,31,28,29]
    i.step()
    i.regfile[calcc]=[0b00000001 for x in range(0,32)]
    i.step()
    i.step()
    i.regfile[calcc]=[0b01111111 for x in range(0,32)]
    i.step()
    i.regfile[calcc]=[0b10000000 for x in range(0,32)]
    i.step()
    i.step()
    i.step()


    ##############################################################################
    i.regfile[calcc]=[3,0,1,2,  7,4,5,6, 11,8,9,10, 15,12,13,14, 19,16,17,18, 23,20,21,22, 27,24,25,26, 31,28,29,30]

    i.step()
    i.regfile[calcc]=[0b00000010 for x in range(0,32)]
    i.step()
    i.step()
    i.regfile[calcc]=[0b00111111 for x in range(0,32)]
    i.step()
    i.regfile[calcc]=[0b11000000 for x in range(0,32)]
    i.step()
    i.step()

    i.regfile[calcc]=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    i.step()
    i.step()
    
        ##############################################################################

    i.regfile[calcc]=[1,2,3,0, 5,6,7,4, 9,10,11,8, 13,14,15,12, 17,18,19,16, 21,22,23,20, 25,26,27,24, 29,30,31,28]
    i.step()
    i.regfile[calcc]=[2,3,0,1, 6,7,4,5  ,10,11,8,9 ,14,15,12,13 ,18,19,16,17 ,22,23,20,21 ,26,27,24,25 ,30,31,28,29]
    i.step()
    i.regfile[calcc]=[0b00000111 for x in range(0,32)]
    i.step()
    i.step()
    i.regfile[calcc]=[0b00000001 for x in range(0,32)]
    i.step()
    i.regfile[calcc]=[0b11111110 for x in range(0,32)]
    i.step()
    i.step()
    i.regfile[calcc]=[0b11111111 for x in range(0,32)]
    i.regfile[calcc][0]=0b00000000
    i.regfile[calcc][1]=0b00111111
    i.regfile[calcc][4]=0b00000000
    i.regfile[calcc][5]=0b00111111
    i.regfile[calcc][8]=0b00000000
    i.regfile[calcc][9]=0b00111111
    i.regfile[calcc][12]=0b00000000
    i.regfile[calcc][13]=0b00111111
    i.regfile[calcc][16]=0b00000000
    i.regfile[calcc][17]=0b00111111
    i.regfile[calcc][20]=0b00000000
    i.regfile[calcc][21]=0b00111111
    i.regfile[calcc][24]=0b00000000
    i.regfile[calcc][25]=0b00111111
    i.regfile[calcc][28]=0b00000000
    i.regfile[calcc][29]=0b00111111
    i.step()


    i.regfile[calcc]=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    i.step()
    i.step()

    
    
def sigma_zero_asm_part1(p,data,result,calca,calcb,calcc):
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
    #load constant into calcb
    p.and_(calca,calca,calcb)
    p.add8(data,calca,data)
    #load constant into calcb
    p.permute(calcc,data,calcb)
    p.xor(result,result,calcc)
    
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
    #load constant into calcb
    p.and_(data,data,calcb)
    #load constant into calcb
    p.permute(calcc,data,calcb)
    p.xor(result,result,calcc)
    
def sigma_zero_asm_part2(i,data,result,calca,calcb,calcc):
    i.regfile[calcb]=[3,0,1,2, 7,4,5,6, 11,8,9,10, 15,12,13,14, 19,16,17,18, 23,20,21,22, 27,24,25,26, 31,28,29,30]
    i.step()
    i.regfile[calcb]=[0b00000111 for x in range(0,32)]
    i.step()
    i.step()
    i.regfile[calcb]=[0b00000001 for x in range(0,32)]
    i.step()
    i.regfile[calcb]=[0b11111110 for x in range(0,32)]
    i.step()
    i.step()
    i.step()


    ##############################################################################
    i.regfile[calcb]=[2,3,0,1, 6,7,4,5  ,10,11,8,9 ,14,15,12,13 ,18,19,16,17 ,22,23,20,21 ,26,27,24,25 ,30,31,28,29]
    i.step()
    i.regfile[calcb]=[3,0,1,2,  7,4,5,6, 11,8,9,10, 15,12,13,14, 19,16,17,18, 23,20,21,22, 27,24,25,26, 31,28,29,30]
    i.step()
    i.regfile[calcb]=[0b00000011 for x in range(0,32)]
    i.step()
    i.step()
    i.regfile[calcb]=[0b00011111 for x in range(0,32)]
    i.step()
    i.regfile[calcb]=[0b11100000 for x in range(0,32)]
    i.step()
    i.step()
    i.regfile[calcb]=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    i.step()
    i.step()
    
    
        ##############################################################################

    i.regfile[calcb]=[1,2,3,0, 5,6,7,4, 9,10,11,8, 13,14,15,12, 17,18,19,16, 21,22,23,20, 25,26,27,24, 29,30,31,28]
    i.step()
    i.regfile[calcb]=[2,3,0,1, 6,7,4,5  ,10,11,8,9 ,14,15,12,13 ,18,19,16,17 ,22,23,20,21 ,26,27,24,25 ,30,31,28,29]
    i.step()
    i.regfile[calcb]=[0b00000001 for x in range(0,32)]
    i.step()
    i.step()
    i.regfile[calcb]=[0b01111111 for x in range(0,32)]
    i.step()
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
    i.regfile[calcb]=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    i.step()
    i.step()
    
