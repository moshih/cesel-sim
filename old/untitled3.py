# -*- coding: utf-8 -*-
"""
Created on Wed May 17 00:39:37 2017

@author: mshih
"""

p.permute(calca,calcb,data)
    #set set calcb as 2
    p.permute(data,calcb,data)
    p.ror(data,calcb,data)
    p.ror(calca,calcb,calca)
    #set set calcb as 0b00111111
    p.and_(data,data,calcb)
    #set set calcb as 0b11000000
    p.and_(calca,calca,calcb)
    p.add8(data,calca,data)
    p.and_(result,data,data)


    p.permute(calca,calcb,data)
    #set set calcb as 2
    #p.permute(data,calcb,data)
    #set set calcb as 2
    p.ror(data,calcb,data)
    p.ror(calca,calcb,calca)
    #set set calcb as 0b00111111
    p.and_(data,data,calcb)
    #set set calcb as 0b11000000
    p.and_(calca,calca,calcb)
    p.add8(data,calca,data)
    p.permute(calcc,calcb,data)
    p.xor(result,result,calcc)
    
    p.permute(calca,calcb,data)
    #set set calcb as 2
    p.permute(data,calcb,data)
    #set set calcb as 2
    p.ror(data,calcb,data)
    p.ror(calca,calcb,calca)
    #set set calcb as 0b00111111
    p.and_(data,data,calcb)
    #set set calcb as 0b11000000
    p.and_(calca,calca,calcb)
    p.add8(data,calca,data)
    p.permute(calcc,calcb,data)
    p.xor(result,result,calcc)