# -*- coding: utf-8 -*-
"""
Created on Tue May 16 21:40:24 2017

@author: mshih
"""

    
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
