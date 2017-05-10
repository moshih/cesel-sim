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
    return (op << 12) | (r1 << 8) | (r0 << 4) | (rw << 0)

def decode(instr):
    # Check if it's an R16-type instruction
    assert (instr & 0xF000) >> 12 != 0xF, "can only decode R-type instructions"

    # Simple decode
    op = (instr & 0xF000) >> 12
    rw = (instr & 0x000F) >> 0
    r1 = (instr & 0x0F00) >> 8
    r0 = (instr & 0x00F0) >> 4

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
        vw[0]=v1[v0]
        #for x in range(0,32):
        #    vw[0][x]=v1[v0[x]]
        #print(vw)
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



def main(args=None):
    # Simple proxies to make our code prettier
    # The "Program" class only need integers
    R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15 = np.arange(16)

    # Program Definition
    p = Program()
    p.permute(R0,R1,R2)
    #p.xor(R0, R0, R0)
    #p.add8(R0, R0, R0)
    #p.mul8(R1, R1, R0)

    # Interpreter
    i = Interpreter(program=p, regfile={i: i for i in range(16)})
    
    for x in range(0,32):
        i.regfile[1][x]=(x+2)%32;
        i.regfile[2][x]=x%4;
        
    print("intial states")
    print(i.regfile[0])
    print(i.regfile[1])
    print(i.regfile[2])
    i.step()
    print(i.regfile[0])
    #i.step()
    
#make l can be is 2**64-1
def deriv_k(l):
    return ((448+512)-(l+1))%512;

def total_bits(l):
    return l-l%512+512;
def total_array_len(l):
    return int((l-l%512+512)/8);

def preprocessing(message_array, length):
    length = int(length);
    M=np.zeros(total_array_len(length), dtype=np.uint8);  
    for i in range(int(0), int((length+7)/8)):
        M[i]=message_array[i];
    M[int(length/8)]=M[int(length/8)]+2**(7-length%8);
    
    for i in range(0,8):
        M[M.size-1-i]=(length&(255<<(8*i)))>>(8*i);
    return M

H_00=0x6a09e667
H_10=0xbb67ae85
H_20=0x3c6ef372
H_30=0xa54ff53a
H_40=0x510e527f
H_50=0x9b05688c
H_60=0x1f83d9ab
H_70=0x5be0cd19

#one block is  512 bits, or 64 8 bit blocks or two lanes.
#one_block is assumed to be 64 uint8 array.
#each word is 32 bits, which occupies 4 values
def sha_256_one_block(one_block):
    W=np.zeros(total_array_len(128), dtype=np.uint8);  
    
    #loading in the first 16 words =16*4=64
    for x in range(0,64):
        W[x]=one_block[x]
    
    
    R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15 = np.arange(16)
    p = Program()
    
    i = Interpreter(program=p, regfile={i: i for i in range(16)})

#sigma_zero operation on R0
#operating on 32 bit works (4 values)
def sigma_zero():
    #use R0,R1,R2
    R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15 = np.arange(16)

    # Program Definition
    p = Program()
    
    
    ##############################################################################
    #rotate by 28 to get the first part of sigmazero
    
    
    p.permute(R1,R15,R0)
    #set set R15 as 2
    p.ror(R0,R15,R0)
    p.ror(R1,R15,R1)
    #set set R15 as 0b00111111
    p.and_(R0,R0,R15)
    #set set R15 as 0b11000000
    p.and_(R1,R1,R15)
    p.add8(R0,R1,R0)


    p.permute(R1,R15,R0)
    #set set R15 as 2
    p.permute(R0,R15,R0)
    #set set R15 as 2
    p.ror(R0,R15,R0)
    p.ror(R1,R15,R1)
    #set set R15 as 0b00111111
    p.and_(R0,R0,R15)
    #set set R15 as 0b11000000
    p.and_(R1,R1,R15)
    p.add8(R0,R1,R0)
    
    p.permute(R1,R15,R0)
    #set set R15 as 2
    p.permute(R0,R15,R0)
    #set set R15 as 2
    p.ror(R0,R15,R0)
    p.ror(R1,R15,R1)
    #set set R15 as 0b00111111
    p.and_(R0,R0,R15)
    #set set R15 as 0b11000000
    p.and_(R1,R1,R15)
    p.add8(R0,R1,R0)


    
     ########-------------------------------------------------------------------------------
    # Interpreter
    i = Interpreter(program=p, regfile={i: i for i in range(16)})
    for x in range(0,32):
        i.regfile[0][x]=(x+0b10101010);
    print ([np.binary_repr(n, width=8) for n in i.regfile[0][0:4]])
    ##############################################################################
    #rotate by 28 to get the first part of sigmazero
    i.regfile[15]=[3,0,1,2, 7,4,5,6, 11,8,9,10, 15,12,13,14, 19,16,17,18, 23,20,21,22, 27,24,25,26, 31,28,29,30]
    i.step()
    i.regfile[15]=[0b00000010 for x in range(0,32)]
    i.step()
    i.step()
    #i.regfile[15]=[0b00111111 for x in range(0,32)]
    i.regfile[15]=[0b00111111 for x in range(0,32)]
    i.step()
    #i.regfile[15]=[0b11000000 for x in range(0,32)]
    i.regfile[15]=[0b11000000 for x in range(0,32)]
    i.step()
    i.step()
    print ([np.binary_repr(n, width=8) for n in i.regfile[0][0:4]])
    sigmazero_one=i.regfile[0][0:4]


    ##############################################################################
    #4,5,6,7, 8,9,10,11, 12,13,14,15, 16,17,18,19, 20,21,22,23, ,24,25,26,27, 28,29,30,31
    #rotate by 28 to get the first part of sigmazero
    i.regfile[15]=[2,3,0,1, 6,7,4,5  ,10,11,8,9 ,14,15,12,13 ,18,19,16,17 ,22,23,20,21 ,26,27,24,25 ,30,31,28,29]
    i.step()
    i.regfile[15]=[3,0,1,2,  7,4,5,6, 11,8,9,10, 15,12,13,14, 19,16,17,18, 23,20,21,22, 27,24,25,26, 31,28,29,30]
    i.step()
    i.regfile[15]=[0b00000011 for x in range(0,32)]
    i.step()
    i.step()
    #i.regfile[15]=[0b00111111 for x in range(0,32)]
    i.regfile[15]=[0b00011111 for x in range(0,32)]
    i.step()
    #i.regfile[15]=[0b11000000 for x in range(0,32)]
    i.regfile[15]=[0b11100000 for x in range(0,32)]
    i.step()
    i.step()
    print ([np.binary_repr(n, width=8) for n in i.regfile[0][0:4]])
    sigmazero_two=i.regfile[0][4:8]
    
        ##############################################################################
    #4,5,6,7, 8,9,10,11, 12,13,14,15, 16,17,18,19, 20,21,22,23, ,24,25,26,27, 28,29,30,31
    #rotate by 28 to get the first part of sigmazero
    i.regfile[15]=[2,3,0,1, 6,7,4,5  ,10,11,8,9 ,14,15,12,13 ,18,19,16,17 ,22,23,20,21 ,26,27,24,25 ,30,31,28,29]
    i.step()
    i.regfile[15]=[3,0,1,2,  7,4,5,6, 11,8,9,10, 15,12,13,14, 19,16,17,18, 23,20,21,22, 27,24,25,26, 31,28,29,30]
    i.step()
    i.regfile[15]=[0b00000001 for x in range(0,32)]
    i.step()
    i.step()
    #i.regfile[15]=[0b00111111 for x in range(0,32)]
    i.regfile[15]=[0b01111111 for x in range(0,32)]
    i.step()
    #i.regfile[15]=[0b11000000 for x in range(0,32)]
    i.regfile[15]=[0b10000000 for x in range(0,32)]
    i.step()
    i.step()
    print ([np.binary_repr(n, width=8) for n in i.regfile[0][0:4]])
    sigmazero_three=i.regfile[0][8:12]

    
    #------------------------------------------------------------
    
    
    #sigma_zero operation on R0
#operating on 32 bit works (4 values)
def sigma_one():
    #use R0,R1,R2
    R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15 = np.arange(16)

    # Program Definition
    p = Program()
    
    

    #rotate by 5 to get the second part of sigmazero (rotate by 39 in total)
    
    #set set R15 as the permutation block (3,0,1,2)
    p.permute(R1,R15,R0)
    #set set R15 as 3
    p.ror(R0,R15,R0)
    p.ror(R1,R15,R1)
    #set set R15 as 0b00011111
    p.and_(R0,R0,R15)
    #set set R15 as 0b11100000
    p.and_(R1,R1,R15)
    p.add8(R0,R1,R0)
    
     ########-------------------------------------------------------------------------------
    # Interpreter
    i = Interpreter(program=p, regfile={i: i for i in range(16)})
    for x in range(0,32):
        i.regfile[0][x]=(x+0b10101010);
    print ([np.binary_repr(n, width=8) for n in i.regfile[0][0:4]])
    ##############################################################################
    #rotate by 28 to get the first part of sigmazero
    i.regfile[15]=[1,2,3,0, 5,6,7,4, 9,10,11,8, 13,14,15,12, 17,18,19,16, 21,22,23,20,  25,26,27,24, 29,30,31,28]

    #rotate by 5 to get the second part of sigmazero (rotate by 39 in total)
    i.regfile[15]=[3,0,1,2, 7,4,5,6, 11,8,9,10, 15,12,13,14, 19,16,17,18, 23,20,21,22, 27,24,25,26, 31,28,29,30]
    i.step()
    i.regfile[15]=[0b00000101 for x in range(0,32)]
    i.step()
    i.step()
    i.regfile[15]=[0b00000111 for x in range(0,32)]

    
    i.step()
    i.regfile[15]=[0b11111000      for x in range(0,32)]
    i.step()
    i.step()
    print ([np.binary_repr(n, width=8) for n in i.regfile[0][0:4]])
    #sigmazero_three=i.regfile[0][8:12]
    
    
def sigma_zero1():
    R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15 = np.arange(16)

    # Program Definition
    p = Program()
    
    #set set R1 as the permutation block (3,0,1,2)
    p.permute(R0,R1,R0)
    #set R2  as 4 for all values
    p.ror(R1,R0,R2)
    #p.ror(R0,R0,R2)

    #set R2 as 55 =0b00001111
    #set R3 as 240 =0b11110000
    p.and_(R1,R0,R3)
    p.and_(R0,R0,R2)


    # Interpreter
    i = Interpreter(program=p, regfile={i: i for i in range(16)})
    for x in range(0,32):
        i.regfile[0][x]=(x+1)%32;
    
    i.regfile[1]=[3,0,1,2, 7,4,5,6, 11,8,9,10, 15,12,13,14, 19,16,17,18, 23,20,21,22, 27,24,25,26, 31,28,29,30]
    #print(i.regfile[0][0:8])   
    #print(i.regfile[1][0:8])
    i.step()
    #print(i.regfile[0][0:8])
    #print(i.regfile[1][0:8])
    
    #print ([np.binary_repr(n, width=8) for n in i.regfile[0]])
    #print(i.regfile[0][0:8])
    #print ([np.binary_repr(n, width=8) for n in i.regfile[0][0:8]])
    i.regfile[2]=[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]
    i.step()
    #i.step()
    print ([np.binary_repr(n, width=8) for n in i.regfile[0][0:8]])
    #print ([np.binary_repr(n, width=8) for n in i.regfile[1][0:8]])
    i.regfile[2]=[55 for x in range(0,32)]
    i.regfile[3]=[240 for x in range(0,32)]
    
    i.step()
    i.step()
    
    print ([np.binary_repr(n, width=8) for n in i.regfile[0][0:8]])
    print ([np.binary_repr(n, width=8) for n in i.regfile[1][0:8]])
    
    #print(i.regfile[0])
    #print ([np.binary_repr(n, width=8) for n in i.regfile[0]])
    #print(i.regfile[1])
    #print(i.regfile[2])
    
    
def test1():
    one_block=np.zeros(total_array_len(64), dtype=np.uint8); 

def test1():
    message=np.zeros(total_array_len(16), dtype=np.uint8)  
    message[0]=255
    message[1]=253
    post=preprocessing(message,16);
    print(post)
    print(post[0:31])
    print(len(post)*8)
    

if __name__ == '__main__':
    #main()
    #test()
    sigma_zero()
    #sigma_one()
