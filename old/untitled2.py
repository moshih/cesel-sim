# -*- coding: utf-8 -*-
"""
Created on Tue May 16 21:43:59 2017

@author: mshih
"""

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
    sumationzero_one=np.copy(i.regfile[data][0:4])
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
    sumationzero_two=np.copy(i.regfile[data][4:8])
    i.regfile[calcb]=[4,5,6,7,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
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
    i.regfile[calcb][data]=0b00011111
    i.step()
    sumationzero_three=np.copy(i.regfile[data][8:12])
    i.regfile[calcb]=[8,9,10,11,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    i.step()
    i.step()
    print ([np.binary_repr(n, width=8) for n in i.regfile[result][0:4]])