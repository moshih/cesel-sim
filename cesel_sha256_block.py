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
import loop_one


from winput import winput
def transfer_two_words_part1(p,rega,offseta,regb,offsetb,calca,calcb):
    p.permute(calca,rega,calca)
    p.and_(calcb,calca,calcb)
    
    p.and_(regb,calca,regb)
    p.add8(regb,regb,calcb )
    
def transfer_two_words_part2(i,rega,offseta,regb,offsetb,calca,calcb):
    
    i.regfile[calca]=[0 for x in range(0,32)]
    i.regfile[calca][offsetb:offsetb+8]=[x for x in range(offseta,offseta+8)]
    i.step()
    
    #print("step 1")
    #print(tovalue(i.regfile[calca]))
    
    
    i.regfile[calcb]=[0 for x in range(0,32)]
    i.regfile[calcb][offsetb:offsetb+8]=[255 for x in range(offsetb,offsetb+8)]
    i.step()
    
    #print("step 2")
    #print(tovalue(i.regfile[calcb]))
    
    i.regfile[calca]=[255 for x in range(0,32)]
    i.regfile[calca][offsetb:32]=[0 for x in range(offsetb, 32)]
    i.step()
    
    #print("step 3")
    #print(tovalue(i.regfile[regb]))
    i.step()
    
    #print("step 4")
    #print(tovalue(i.regfile[regb]))
    #print(" ")
    
def sha_256_one_block():
    #one_block=np.zeros(total_array_len(64), dtype=np.uint8); 
    R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15 = np.arange(16)

    # Program Definition
    p = Program()
    #first load into the first 2 registers, the message block
    
    #put in the 14th and 15th word into the register of 2
    loop_one.loop1_batch_part1(p,R2,R3,R4,R5,R6,R7,winput[0],winput[1])
    
    ##p.add8(R2,R2,R4)
    #now create and add sigma0
    #R0 and R1 have the original data
    #R2 contains W17 and W18
    ########################################################
    loop_one.loop1_batch_part1(p,R3,R4,R5,R6,R7,R8,winput[2],winput[3])
    transfer_two_words_part1(p,R3,0,R2,8,R4,R5)
    
    loop_one.loop1_batch_part1(p,R3,R4,R5,R6,R7,R8,winput[4],winput[5])
    transfer_two_words_part1(p,R3,0,R2,16,R4,R5)
    
    loop_one.loop1_batch_part1(p,R3,R4,R5,R6,R7,R8,winput[6],winput[7])
    transfer_two_words_part1(p,R3,0,R2,24,R4,R5)
    
    ###########################
    loop_one.loop1_batch_part1(p,R4,R5,R6,R7,R8,R9,winput[8],winput[9])
    transfer_two_words_part1(p,R4,0,R3,0,R5,R6)
    
    loop_one.loop1_batch_part1(p,R4,R5,R6,R7,R8,R9,winput[10],winput[11])
    transfer_two_words_part1(p,R4,0,R3,8,R5,R6)
    
    loop_one.loop1_batch_part1(p,R4,R5,R6,R7,R8,R9,winput[12],winput[13])
    transfer_two_words_part1(p,R4,0,R3,16,R5,R6)
    
    loop_one.loop1_batch_part1(p,R4,R5,R6,R7,R8,R9,winput[14],winput[15])
    transfer_two_words_part1(p,R4,0,R3,24,R5,R6)
    
    
     ###########################
    loop_one.loop1_batch_part1(p,R5,R6,R7,R8,R9,R10,winput[16],winput[17])
    transfer_two_words_part1(p,R5,0,R4,0,R6,R7)
    loop_one.loop1_batch_part1(p,R5,R6,R7,R8,R9,R10,winput[18],winput[19])
    transfer_two_words_part1(p,R5,0,R4,8,R6,R7)
    loop_one.loop1_batch_part1(p,R5,R6,R7,R8,R9,R10,winput[20],winput[21])
    transfer_two_words_part1(p,R5,0,R4,16,R6,R7)
    loop_one.loop1_batch_part1(p,R5,R6,R7,R8,R9,R10,winput[22],winput[23])
    transfer_two_words_part1(p,R5,0,R4,24,R6,R7)

     ###########################
    loop_one.loop1_batch_part1(p,R6,R7,R8,R9,R10,R11,winput[24],winput[25])
    transfer_two_words_part1(p,R6,0,R5,0,R7,R8)
    loop_one.loop1_batch_part1(p,R6,R7,R8,R9,R10,R11,winput[26],winput[27])
    transfer_two_words_part1(p,R6,0,R5,8,R7,R8)
    loop_one.loop1_batch_part1(p,R6,R7,R8,R9,R10,R11,winput[28],winput[29])
    transfer_two_words_part1(p,R6,0,R5,16,R7,R8)
    loop_one.loop1_batch_part1(p,R6,R7,R8,R9,R10,R11,winput[30],winput[31])
    transfer_two_words_part1(p,R6,0,R5,24,R7,R8)
    
    ###############################--------
 
    

    loop_one.loop1_batch_part1(p,R7,R8,R9,R10,R11,R12,winput[32],winput[33])
    transfer_two_words_part1(p,R7,0,R6,0,R8,R9)                              
    loop_one.loop1_batch_part1(p,R7,R8,R9,R10,R11,R12,winput[34],winput[35]) 
    transfer_two_words_part1(p,R7,0,R6,8,R8,R9)                              
    loop_one.loop1_batch_part1(p,R7,R8,R9,R10,R11,R12,winput[36],winput[37]) 
    transfer_two_words_part1(p,R7,0,R6,16,R8,R9)                             
    loop_one.loop1_batch_part1(p,R7,R8,R9,R10,R11,R12,winput[38],winput[39]) 
    transfer_two_words_part1(p,R7,0,R6,24,R8,R9)


    loop_one.loop1_batch_part1(p,R8,R9,R10,R11,R12,R13,winput[40],winput[41])
    transfer_two_words_part1(p,R8,0,R7,0,R9,R10)                             
    loop_one.loop1_batch_part1(p,R8,R9,R10,R11,R12,R13,winput[42],winput[43])
    transfer_two_words_part1(p,R8,0,R7,8,R9,R10)                             
    loop_one.loop1_batch_part1(p,R8,R9,R10,R11,R12,R13,winput[44],winput[45])
    transfer_two_words_part1(p,R8,0,R7,16,R9,R10)                            
    loop_one.loop1_batch_part1(p,R8,R9,R10,R11,R12,R13,winput[46],winput[47])
    transfer_two_words_part1(p,R8,0,R7,24,R9,R10)



    
    ########################################################################################################################################################################
    # Interpreter
    i = Interpreter(program=p, regfile={i: i for i in range(16)})
    
    for x in range(0,32):
        i.regfile[0][x]=x
        i.regfile[1][x]=x+32
    
    
    check=0;
    print("W 1st row")
    print(tovalue(i.regfile[0]))
    print("W 2bd row")
    print(tovalue(i.regfile[1]))
    
    loop_one.loop1_batch_part2(i,R2,R3,R4,R5,R6,R7,winput[0],winput[1])
    
    ####################################
    #i.regfile[2][8:16]=[0,0,0,1 ,01111101, 00110000, 00011011, 01101001]#[11110001, 00011001, 11110111, 00110001,01101001 ,00011011 ,00110000,01111101]
    #[00110001, 11110111, 00011001, 11110001 ,01111101, 00110000, 00011011, 01101001]
    
    #print("test")
    #print(tovalue(i.regfile[2]))
    #i.regfile[2][8:12]=[00110001, 11110111, 00011001, 11110001]
    #i.regfile[2][12:16]=[01111101 ,00110000 ,00011011 ,01101001]
    loop_one.loop1_batch_part2(i,R3,R4,R5,R6,R7,R8,winput[2],winput[3])
    transfer_two_words_part2(i,R3,0,R2,8,R4,R5)
    #print("test")
    #print(tovalue(i.regfile[2]))
    
    loop_one.loop1_batch_part2(i,R3,R4,R5,R6,R7,R8,winput[4],winput[5])
    transfer_two_words_part2(i,R3,0,R2,16,R4,R5)
    #print("test")
    #print(tovalue(i.regfile[2]))
    
    loop_one.loop1_batch_part2(i,R3,R4,R5,R6,R7,R8,winput[6],winput[7])
    transfer_two_words_part2(i,R3,0,R2,24,R4,R5)
    
    
    print("W 3rd row")
    print(tovalue(i.regfile[2]))
    
    loop_one.loop1_batch_part2(i,R4,R5,R6,R7,R8,R9,winput[8],winput[9])
    transfer_two_words_part2(i,R4,0,R3,0,R5,R6)
    loop_one.loop1_batch_part2(i,R4,R5,R6,R7,R8,R9,winput[10],winput[11])
    transfer_two_words_part2(i,R4,0,R3,8,R5,R6)
    loop_one.loop1_batch_part2(i,R4,R5,R6,R7,R8,R9,winput[12],winput[13])
    transfer_two_words_part2(i,R4,0,R3,16,R5,R6)
    loop_one.loop1_batch_part2(i,R4,R5,R6,R7,R8,R9,winput[14],winput[15])
    transfer_two_words_part2(i,R4,0,R3,24,R5,R6)
    
    
    print("W 4th row")
    print(tovalue(i.regfile[3]))
    

    loop_one.loop1_batch_part2(i,R5,R6,R7,R8,R9,R10,winput[16],winput[17])
    transfer_two_words_part2(i,R5,0,R4,0,R6,R7)
    loop_one.loop1_batch_part2(i,R5,R6,R7,R8,R9,R10,winput[18],winput[19])
    print(tovalue(i.regfile[5]))
    transfer_two_words_part2(i,R5,0,R4,8,R6,R7)
    loop_one.loop1_batch_part2(i,R5,R6,R7,R8,R9,R10,winput[20],winput[21])
    transfer_two_words_part2(i,R5,0,R4,16,R6,R7)
    loop_one.loop1_batch_part2(i,R5,R6,R7,R8,R9,R10,winput[22],winput[23])
    transfer_two_words_part2(i,R5,0,R4,24,R6,R7)

    
    print("W 5th row")
    print(tovalue(i.regfile[4]))
    
    
    
    loop_one.loop1_batch_part2(i,R6,R7,R8,R9,R10,R11,winput[24],winput[25])
    transfer_two_words_part2(i,R6,0,R5,0,R7,R8)
    loop_one.loop1_batch_part2(i,R6,R7,R8,R9,R10,R11,winput[26],winput[27])
    transfer_two_words_part2(i,R6,0,R5,8,R7,R8)
    loop_one.loop1_batch_part2(i,R6,R7,R8,R9,R10,R11,winput[28],winput[29])
    transfer_two_words_part2(i,R6,0,R5,16,R7,R8)
    loop_one.loop1_batch_part2(i,R6,R7,R8,R9,R10,R11,winput[30],winput[31])
    transfer_two_words_part2(i,R6,0,R5,24,R7,R8)


    
    print("W 6th row")
    print(tovalue(i.regfile[5]))
    
    loop_one.loop1_batch_part2(i,R7,R8,R9,R10,R11,R12,winput[32],winput[33])
    transfer_two_words_part2(i,R7,0,R6,0,R8,R9)
    loop_one.loop1_batch_part2(i,R7,R8,R9,R10,R11,R12,winput[34],winput[35])
    transfer_two_words_part2(i,R7,0,R6,8,R8,R9)
    loop_one.loop1_batch_part2(i,R7,R8,R9,R10,R11,R12,winput[36],winput[37])
    transfer_two_words_part2(i,R7,0,R6,16,R8,R9)
    loop_one.loop1_batch_part2(i,R7,R8,R9,R10,R11,R12,winput[38],winput[39])
    transfer_two_words_part2(i,R7,0,R6,24,R8,R9)
    print("W 7th row")
    print(tovalue(i.regfile[6]))


    loop_one.loop1_batch_part2(i,R8,R9,R10,R11,R12,R13,winput[40],winput[41])
    transfer_two_words_part2(i,R8,0,R7,0,R9,R10)
    loop_one.loop1_batch_part2(i,R8,R9,R10,R11,R12,R13,winput[42],winput[43])
    transfer_two_words_part2(i,R8,0,R7,8,R9,R10)
    loop_one.loop1_batch_part2(i,R8,R9,R10,R11,R12,R13,winput[44],winput[45])
    transfer_two_words_part2(i,R8,0,R7,16,R9,R10)
    loop_one.loop1_batch_part2(i,R8,R9,R10,R11,R12,R13,winput[46],winput[47])
    transfer_two_words_part2(i,R8,0,R7,24,R9,R10)
    print("W 8th row")
    print(tovalue(i.regfile[7]))


    #print("data")
    #print(tovalue(i.regfile[0]))
    #print(tovalue(i.regfile[1]))
    #print(tovalue(i.regfile[2]))

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