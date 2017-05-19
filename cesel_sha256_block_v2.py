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
EX_OP_PERMUTE_constant   = 0b1011

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
    elif op == EX_OP_PERMUTE_constant:
        #vw = v1 + v0
        vw[0]=v0[v1]
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
    def permutec(self, rw, r1, r0):
        self.code.append(instr_r16(EX_OP_PERMUTE_constant, rw, r1, r0))

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

def sha_256_one_block():
    #one_block=np.zeros(total_array_len(64), dtype=np.uint8); 
    R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15 = np.arange(16)

    # Program Definition
    p = Program()
    #first load into the first 2 registers, the message block
    
    #put in the 14th and 15th word into the register of 2
    p.permute(R2,R1,R3)
    #now we apply sigma_one
    #R2 is the data
    #R4 is used for calcuations
    #R5 is used for calcuations
    #R3 loads the result
    #R6 is used for calculation
    data=R2
    calca=R4
    calcb=R5
    calc=R6
    result=R3
    
    #p.permute(R4,R2,R6)
    sigma_one_asm_part1(p,R2,R3,R4,R5,R6)
    
    
    
    #R0 and R1 have the original data
    #R2 is the sigma1 of the next 2
    p.permute(R2,R0,R2)
    ##p.add8(R2,R2,R3)
    p.permute(R3,R1,R3)
    ##p.add8(R2,R2,R3)
    #now create and add sigma0
    #R0 and R1 have the original data
    #R2 is the sigma1 +2 things 
    
    asm_func.sigma_zero_asm_part1(p, R3,R4,R5,R6,R7)
    
    ##p.add8(R2,R2,R4)
    #now create and add sigma0
    #R0 and R1 have the original data
    #R2 contains W17 and W18
    ########################################################
    p.permute(R4,R2,R4)
    sigma_one_asm_part1(p,R4,R3,R5,R6,R7)
    
    
    p.permute(R4,R1,R4)
    p.and_(R4,R4,R5)
    
    p.permute(R5,R2,R5)
    p.and_(R5,R5,R6)
    p.add8(R4,R4,R5)
    
    p.permute(R5,R0,R5)
    p.and_(R5,R5,R6)
    
    p.permute(R6,R1,R6)
    p.and_(R6,R6,R7)
    p.add8(R5,R5,R6)
    
    ##add_32 bit word R4, R4,R5
    ##add_32 bit word R3,R3,R4
    
    #sigma_zero
    p.permute(R4,R0,R4)
    p.and_(R4,R4,R5)
    
    p.permute(R5,R1,R5)
    p.and_(R5,R5,R6)
    p.add8(R4,R4,R5)
    asm_func.sigma_zero_asm_part1(p, R4,R5,R6,R7,R8)
    
    ########################################################################################################################################################################
    # Interpreter
    i = Interpreter(program=p, regfile={i: i for i in range(16)})
    
    for x in range(0,32):
        i.regfile[0][x]=x
        i.regfile[1][x]=x+32
    i.regfile[R3]=[ 27,26,25,24,  31,30,29,28, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,  0,0,0,0, 0,0,0,0]
    i.step()
    
    sigma_one_asm_part2(i,R2,R3,R4,R5,R6)
    
    
    i.regfile[2]=get_index(0,0)+get_index(0,1)+[ 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0  ]
    i.step()
    i.regfile[2]=add(i.regfile[2],i.regfile[3])
    
    ##i.step()
    i.regfile[3]=get_index(1,1)+get_index(1,2)+[ 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0  ]
    i.step()
    
    
    ##i.step()
    i.regfile[2]=add(i.regfile[2],i.regfile[3])
    
    i.regfile[3]=get_index(0,1)+get_index(0,2)+[ 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0  ]
    asm_func.sigma_zero_asm_part2(i, R3,R4,R5,R6,R7)
    

   ## i.step()
    i.regfile[2]=add(i.regfile[2],i.regfile[4])
    print("0th bath of size 2")
    print(tovalue(i.regfile[2]))
    
    
    ####################################
    
    i.regfile[4]=get_index(2,0)+get_index(2,1)+get_index(2,2)+get_index(2,3)+get_index(2,4)+get_index(2,5)+get_index(2,6)+get_index(2,7)
    i.step()
    #print(tovalue(i.regfile[4]))
    sigma_one_asm_part2(i,R4,R3,R5,R6,R7)
    #print("sigma1 of full batch 1")
    #print(tovalue(i.regfile[3]))
    
    ### At this point, R3 holds sigma 1 for this batch (first full batch)
    
    i.regfile[4]=get_index(1,3)+get_index(1,4)+get_index(1,5)+get_index(1,6)+get_index(1,7)+[ 0,0,0,0, 0,0,0,0, 0,0,0,0]
    i.step()
    i.regfile[5]=[255,255,255,255, 255,255,255,255, 255,255,255,255, 255,255,255,255, 255,255,255,255, 0,0,0,0, 0,0,0,0, 0,0,0,0 ]
    i.step()
    #print(tovalue(i.regfile[4]))
    i.regfile[5]=[ 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]+get_index(2,0)+get_index(2,1)+get_index(2,2)
    i.step()
    i.regfile[6]=[0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,  255,255,255,255, 255,255,255,255, 255,255,255,255 ]
    i.step()
    #print(tovalue(i.regfile[5]))
    i.step()
    #print(tovalue(i.regfile[4]))
    
    i.regfile[5]=get_index(0,2)+get_index(0,3)+get_index(0,4)+get_index(0,5)+get_index(0,6)+get_index(0,7)+[  0,0,0,0, 0,0,0,0]
    i.step()
    i.regfile[6]=[255,255,255,255, 255,255,255,255, 255,255,255,255, 255,255,255,255, 255,255,255,255, 255,255,255,255, 0,0,0,0, 0,0,0,0 ]
    i.step()
    #print(tovalue(i.regfile[4]))
    #print(tovalue(i.regfile[5]))
    
    i.regfile[6]=[ 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]+get_index(1,0)+get_index(1,1)
    i.step()
    i.regfile[7]=[0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,  0,0,0,0, 255,255,255,255, 255,255,255,255 ]
    i.step()
    #print(tovalue(i.regfile[6]))
    #print(tovalue(i.regfile[5]))
    i.step()
    #print(tovalue(i.regfile[5]))
    ##i.step()
    i.regfile[4]=add(i.regfile[4],i.regfile[5])
    #print("some of two of full batch 1")
    #print(tovalue(i.regfile[4]))
    i.regfile[3]=add(i.regfile[3],i.regfile[4])
    
    
    i.regfile[4]=get_index(0,3)+get_index(0,4)+get_index(0,5)+get_index(0,6)+get_index(0,7)+[ 0,0,0,0, 0,0,0,0, 0,0,0,0]
    i.step()
    i.regfile[5]=[255,255,255,255, 255,255,255,255, 255,255,255,255, 255,255,255,255, 255,255,255,255, 0,0,0,0, 0,0,0,0, 0,0,0,0 ]
    i.step()
    #print(tovalue(i.regfile[4]))
    i.regfile[5]=[ 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]+get_index(1,0)+get_index(1,1)+get_index(1,2)
    i.step()
    i.regfile[6]=[0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,  255,255,255,255, 255,255,255,255, 255,255,255,255 ]
    i.step()
    #print(tovalue(i.regfile[5]))
    i.step()
    #print(tovalue(i.regfile[4]))
    asm_func.sigma_zero_asm_part2(i, R4,R5,R6,R7,R8)
    #print(tovalue(i.regfile[5]))
    #print(tovalue(i.regfile[3]))
    i.regfile[3]=add(i.regfile[3],i.regfile[5])
    print("finished of full batch 1")
    print(tovalue(i.regfile[3]))
        

def test123():
    for x in range(16,64):
        a=x-2
        b=x-7
        c=x-15
        d=x-16
        
        a1=int((a-a%8)/8)
        a2=a%8
        b1=int((b-b%8)/8)
        b2=b%8
        c1=int((c-c%8)/8)
        c2=c%8
        d1=int((d-d%8)/8)
        d2=d%8
        print(a1,a2, "|",b1,b2, "|",c1,c2, "|",d1,d2)
from cesel_extra import sigma_zero
if __name__ == '__main__':
    #sha_a()
    #main()
    #test123()
    sha_256_one_block()
    #sumation_zero()
    #sumation_one()
    #sigma_zero()
    #sigma_one()