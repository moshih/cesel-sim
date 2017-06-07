# -*- coding: utf-8 -*-
"""
Created on Mon Jun 05 17:57:05 2017

@author: mshih
"""

from __future__ import division
from __future__ import print_function

import numpy as np


# Instruction constants
INSTR_BITS = 16

# Total number of lanes
N_LANES = 32

# Registerfile constants
REGFILE_N_REGS    = 16
REGFILE_DATA_BITS = 8
REGFILE_ADDR_BITS = np.uint(np.ceil(np.log2(REGFILE_N_REGS)))

# EX stage constants
EX_OP_BITS     = 4

EX_OP_XOR       = 0b0000
EX_OP_AND       = 0b0001
EX_OP_XNOR      = 0b0010
EX_OP_ROR       = 0b0011
EX_OP_ADD       = 0b0100
EX_OP_SUB       = 0b0101
EX_OP_MUL       = 0b0110
EX_OP_GF2       = 0b0111
EX_OP_BITSLICE  = 0b1000
EX_OP_PERMUTE   = 0b1001
EX_OP_SBOX      = 0b1010
EX_OP_RESERVED1 = 0b1100 # Load immediate
EX_OP_RESERVED2 = 0b1101
EX_OP_RESERVED3 = 0b1110
EX_OP_RESERVED4 = 0b1111 # Extended Instr
EX_OP_SUM   = 0b1011

# Rotate/shift
# Load immediate
# Pop Message/Key
# Push output
# Bitmatmul?
# wide mult/add/sub?
# Loops?
# Masking?

def instr_r16(op, rw, r1, r0):
    """Insert a 16-bit R-type instruction"""
    assert op & 0xFF == op, "op must be an 8-bit value"
    assert rw & 0xFF == rw, "rw must be an 8-bit value"
    assert r1 & 0xFF == r1, "r1 must be an 8-bit value"
    assert r0 & 0xFF == r0, "r0 must be an 8-bit value"

    # R16-type instruction
    # 4 bits | 4 bits | 4 bits | 4 bits
    # Ex Op  | r1     | r0     | rw
    return (op << 24) | (r1 << 16) | (r0 << 8) | (rw << 0)

def decode(instr):
    # Check if it's an R16-type instruction
    assert (instr & 0xF000) >> 12 != 0xF, "can only decode R-type instructions"

    # Simple decode
    op = (instr & 0xFF000000) >> 24
    rw = (instr & 0x000FF) >> 0
    r1 = (instr & 0x0FF0000) >> 16
    r0 = (instr & 0x00FF00) >> 8

    # Write is always enabled for now
    write_en = 1

    return op, rw, write_en, r1, r0

def execute(op, v1, v0, acc):
    vw = np.zeros(shape=acc.shape, dtype=acc.dtype)
    if op == EX_OP_XOR:
        vw = v1 ^ v0
    elif op == EX_OP_AND:
        vw = v1 & v0
    elif op == EX_OP_XNOR:
        vw = ~(v1 ^ v0)
    elif op == EX_OP_ROR:
        vw = (v1 >> v0) | (v1 << 8 - v0)
    elif op == EX_OP_ADD:
        vw = v1 + v0
    elif op == EX_OP_SUB:
        vw = v1 - v0
    elif op == EX_OP_MUL:
        vw = v1 * v0
    elif op == EX_OP_BITSLICE:
        # Seperate v1 into 4 groups of 8 bits
        for i in range(4):
            for j in range(8):
                # We want to collect the kth bit of each element in v1[i*8 + j]
                # Create a mask we can apply to each byte
                mask = 0x01 << j
                for k in range(8):
                    vw[i*8 + j] |= ((v1[i*8 + k] & mask) >> i) << k
        # TODO
    elif op == EX_OP_PERMUTE:
        # For each value in the shuffle mask check the top bits for "special
        # values"
        
        for i, v in enumerate(v0):
            # If any of the top 3 bits are set, set the register to "special value"
            vspecial = (v & 0xe) >> 5
            if vspecial:
                if vspecial == 0b111:
                    v1[i] = 0xFF
                elif vspecial == 0b100:
                    v1[i] = 0x00
                elif vspecial == 0b101:
                    v1[i] = 0x01
                elif vspecial == 0b110:
                    v1[i] = i

        # Shuffle the bytes based on the indexes in v0
        #print(vw)
        #print(v0)
        #print(v1)
        #vw[v0] = v1
        #print(vw)
        vw[0]=v0[v1]
        #for x in range(0,32):
        #    vw[0][x]=v1[v0[x]]
        #print(vw)
    elif op == EX_OP_SUM:
        #vw = v1 + v0
        total=0;
        for x in range(0,16):
            total=total+256*v0[2*x]+v0[2*x+1]
        for x in range(0,16):
            vw[0][2*x+1]=total 
       
    else:
        raise Exception("Illegal Instruction!")

    return vw, acc

class Program(object):
    """Container for the asm program"""
    def __init__(self, name='<unnamed>', code=[]):
        self.name = name
        self.code = code

    def xor(self, rw, r1, r0):
        self.code.append(instr_r16(EX_OP_XOR, rw, r1, r0))

    def and_(self, rw, r1, r0):
        self.code.append(instr_r16(EX_OP_AND, rw, r1, r0))

    def xnor(self, rw, r1, r0):
        self.code.append(instr_r16(EX_OP_XNOR, rw, r1, r0))

    def ror(self, rw, r1, r0):
        self.code.append(instr_r16(EX_OP_ROR, rw, r1, r0))

    def add8(self, rw, r1, r0):
        self.code.append(instr_r16(EX_OP_ADD, rw, r1, r0))

    def sub8(self, rw, r1, r0):
        self.code.append(instr_r16(EX_OP_SUB, rw, r1, r0))

    def mul8(self, rw, r1, r0):
        self.code.append(instr_r16(EX_OP_MUL, rw, r1, r0))

    def bitslice(self, rw, r1, r0):
        self.code.append(instr_r16(EX_OP_BITSLICE, rw, r1, r0))

    def permute(self, rw, r1, r0):
        self.code.append(instr_r16(EX_OP_PERMUTE, rw, r1, r0))
    def sum_reg(self, rw, r1, r0):
        self.code.append(instr_r16(EX_OP_SUM, rw, r1, r0))

    def __str__(self):
        return "Program(name='{}')".format(self.name)

    def serialize(self, fmt="{:04x}\n"):
        result = ""
        for instr in self.code:
            result += fmt.format(instr)

        return result

class Interpreter(object):
    """Simple interpreter for a program"""
    def __init__(self, program=None, state="running", pc=0, regfile=None):
        # Default to an empty program
        if program is None:
            program = Program()

        # Code
        self.program = program

        # Current instruction
        self.pc = pc

        # Register file
        # several helpers to setup register file
        regfile_shape = (REGFILE_N_REGS, N_LANES)
        if regfile is None:
            # Default to all zeros
            self.regfile = np.zeros(shape=regfile_shape, dtype=np.uint8)

        elif regfile is 'random' or regfile is 'rand':
            # Random initialization
            self.regfile = np.random.randint(256, size=regfile_shape, dtype=np.uint8)

        elif isinstance(regfile, int):
            # All values equal
            self.regfile = np.full(shape=regfile_shape, fill_value=regfile, dtype=np.uint8)

        elif isinstance(regfile, dict):
            # Set specific registers equal to value using a dict
            # e.g. {0: 0xFF, 1: 0x11} sets register 0 to 0xFF and register 1 to 0x11
            self.regfile = np.zeros(shape=regfile_shape, dtype=np.uint8)
            for k, v in regfile.viewitems():
                self.regfile[k] = v

        else:
            # Just convert whatever crap they gave us to a numpy array
            self.regfile = np.asarray(regfile, dtype=uint8, copy=True)

            # Attempt to at least convert it to the right shape
            self.regfile = np.broadcast_to(self.regfile, regfile_shape)

        # Accumulator State
        self.acc = np.zeros(shape=(1, N_LANES), dtype=np.uint16)

        # Current "State" (one of "running" or "halted")
        self.state = state

    def step(self):
        # Get the current instruction
        instr = self.program.code[self.pc]

        # Decode the instruction
        op, rw, write_en, r1, r0 = decode(instr)

        # Read the regfile
        v0 = self.regfile[r0]
        v1 = self.regfile[r1]

        # Execute
        try:
            vw, self.acc = execute(op, v0, v1, self.acc)
        except RuntimeError as e:
            raise RuntimeError("Exception in {} at pc = {}: {}"
                    .format(self.program, self.pc, e))

        # Writeback
        if write_en:
            self.regfile[rw] = vw

        # Advance to next instruction
        self.pc += 1

        return self.state

    def run(self):
        while self.state != "halted":
            self.step()

        return self.state

    def __str__(self):
        return "Interpreter(state='{}', pc={})".format(self.state, self.pc)

np.set_printoptions(threshold=np.nan)

def print_mat(A):
    for x in range(0,8):
        print(A[0][x],A[1][x],A[2][x],A[3][x],A[4][x],A[5][x],A[6][x],A[7][x])
def mul16_part1():
    2
def mul16_part2(i,rega,regb):
    result=i.regfile[rega];
    for x in range(0,16):
        product=(256*i.regfile[rega][2*x]+i.regfile[rega][2*x+1])*(256*i.regfile[regb][2*x]+i.regfile[regb][2*x+1])
        #print(product,(product&(255<<8))>>8,product%256)
        result[2*x]=(product&(255<<8))>>8
        result[2*x+1]=product%256
    return result

def add16_part1():
    2
def add16_part2(i,rega,regb):
    result=i.regfile[rega];
    for x in range(0,16):
        product=(256*i.regfile[rega][2*x]+i.regfile[rega][2*x+1])+(256*i.regfile[regb][2*x]+i.regfile[regb][2*x+1])
        #print(product,(product&(255<<8))>>8,product%256)
        result[2*x]=(product&(255<<8))>>8
        result[2*x+1]=product%256
    return result


#index up to 46
def load_A_row(i,A,reg, row,index):

    for x in range(0,16):
        i.regfile[reg][2*x]=(A[row][index*16+x] &(255<<8))>>8
        i.regfile[reg][2*x+1]=A[row][index*16+x]%256
        
def load_A_col(i,A,reg, col,index):
    for x in range(0,16):
        i.regfile[reg][2*x]=(A[index*16+x][col] &(255<<8))>>8
        i.regfile[reg][2*x+1]=A[index*16+x][col]%256
        
def load_tall_row(i,A,reg, row,index):
    for x in range(0,8):
        i.regfile[reg][2*x]=(A[row][index*16+x] &(255<<8))>>8
        i.regfile[reg][2*x+1]=A[row][index*16+x]%256
    for x in range(0,8):
        i.regfile[reg][2*x]=(A[row+1][index*16+x] &(255<<8))>>8
        i.regfile[reg][2*x+1]=A[row+1][index*16+x]%256
        
def load_tall_col(i,A,reg, col,index):
    for x in range(0,16):
        i.regfile[reg][2*x]=(A[index*16+x][col] &(255<<8))>>8
        i.regfile[reg][2*x+1]=A[index*16+x][col]%256
        
#index up to 46
def load_long_row(i,A,reg, row,index):
    for x in range(0,16):
        i.regfile[reg][2*x]=(A[row][index*16+x] &(255<<8))>>8
        i.regfile[reg][2*x+1]=A[row][index*16+x]%256
        
def load_long_col(i,A,reg, col,index):
    for x in range(0,8):
        i.regfile[reg][2*x]=(A[index*16+x][col] &(255<<8))>>8
        i.regfile[reg][2*x+1]=A[index*16+x][col]%256
    for x in range(8,16):
        i.regfile[reg][2*x]=0
        i.regfile[reg][2*x+1]=0
        
        
def print_reg(reg):
    print([reg[2*x]*256+reg[2*x+1] for x in range(0,16)])
        
def As_mul_part1(p,inputa,inputb,temp,calca):
    mul16_part1()
    p.sum_reg(temp,calca,calca)
    
    for x in range(0,46):
        #load
        #load
        mul16_part1()
        p.sum_reg(inputa,calca,calca)
        add16_part1()
        
def As_mul_part2(i,inputa,inputb,temp,calca,row,col,A,s):
    load_A_row(i,A,inputa,row,0);
    load_tall_col(i,s,inputb,col,0);

    i.regfile[calca]= mul16_part2(i,inputa,0);
    i.step();

    
    ####
    for x in range(1,47):
        load_A_row(i,A,inputa,row,x);
        load_tall_col(i,s,inputb,col,x);
        i.regfile[calca]= mul16_part2(i,inputa,inputb);
        #print_reg(i.regfile[inputa]);
        #print_reg(i.regfile[inputb]);
        #print_reg(i.regfile[calca]);
        i.step()
        #print_reg(i.regfile[inputa]);
        
        i.regfile[temp]=add16_part2(i,temp,inputa)
def test():
    A = [[x+y for x in range(0,752)] for y in range(0,752)] 
    s = [[0 for x in range(0,8)] for y in range(0,752)] 
    e = [[0 for x in range(0,8)] for y in range(0,752)] 
    for x in range(0,8):
        for y in range(0,8):
            s[x][y]=1;
            
    b= [[0 for x in range(16)] for y in range(752)] 
    
    

    

    R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15 = np.arange(16)

    # Program Definition
    p = Program()
    
    #############################
    #Matrix multiply
    
    result=R0;
    inputa=R1;
    inputb=R2;
    temp=R3
    calca=R4;
    
   
    #load

    for col in range(0,1 ):
        for index in range(0,47):
            As_mul_part1(p,R1,R2,R3,R4)
            #load
            if col==0:
                p.and_(R0,R1,R3)
            else:
                p.and_(R2,R1,R3)
            #add
    


    #p.sum_reg(R0,R1,R2)
    #######################################################
    # Interpreter
    i = Interpreter(program=p, regfile={i: i for i in range(16)})
    #i.regfile[0]=[0b10101010, 0b10101010, 0b10101011, 0b11101010, 0b10101010, 0b00101010, 0b10101110, 0b11101010, 0b11111010, 0b10101010]+[0 for x in range(0,22)]
    #i.step()
    #print(i.regfile[R0])
    
    #############################
    #Matrix multiply
    load_tall_row(i,e,result,0,0);
    #row=0;
    #col=0;
    #for y in range(0,int(752/2)):

    row=0
    for col in range(0,1):
        As_mul_part2(i,R1,R2,R3,R4,row+int(col/8),col%8,A,s)
        
        i.regfile[R1]=[0 for x in range(0,32)]
        i.regfile[R1][(2*col)%32]=255;
        i.regfile[R1][(2*col+1)%32]=255;
        i.step();
        if col!=0:
            i.regfile[result]=add16_part2(i,result,R2)
        
    
    ###################
    b[row]=i.regfile[result][0:16]
    b[row+1]=i.regfile[result][16:32]
    
    print(i.regfile[result])
    print_reg(i.regfile[result]);
    
    
    #print(A[751])
    
    

    
    #print_reg(i.regfile[result]);
    
        

    

if __name__ == '__main__':
    test()