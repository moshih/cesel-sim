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

# Instruction formats
# | 5 | 4 | 3 | 2 | 1 | 0 | 9 | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0
# | 0 | im        | SO    | rw/r1         | r0            | 0   0
# | 1   0 | im    | AO    | rw/r1         | r0            | 0   0
# | 1   1 | 0   1 | CO    | rw/r1         | r0            | 0   0

# Shift/Add by short constant
# | SO    | 0 | im3       | rw            | r0            | 0   0
# | AO    | 1 | im3       | rw            | r0            | 0   0

# 2 Reg Operations
# | LO    | 0   0 | I | I | rw/r1         | r0            | 0   1
# | SO    | 0   1 | X   X | rw/r1         | r0            | 0   1
# | AO    | 1   0 | X   X | rw/r1         | r0            | 0   1
# | CO    | 1   1 | X   X | rw/r1         | r0            | 0   1

# Wide arithmatic operations
# | AO    | C |           | rw/r1         | r0            | 1   0

# Long and "Special" instructions
# | LDI   | imm4          | rw            | imm4          | 1   1
# | AND   | imm4          | rw            | imm4          | 1   1
# | XOR   | imm4          | rw            | imm4          | 1   1
# | 1   1 | LOAD_LONG     | rw            | size          | 1   1
# | 1   1 | PUSH          | r1            | index         | 1   1
# | 1   1 | PULL          | rw            | index         | 1   1
# | 1   1 | LOOP_BEGIN    | 1   1   1   1 | imm4          | 1   1
# | 1   1 | LOOP_END      | r1            | 1   1   1   1 | 1   1
# | 1   1 | HALT          | 1   1   1   1   1   1   1   1 | 1   1

# Broadcast low16/high16 - Take value in lane N or N + 16 and copy to all lanes
# | 0   0 | rw            | r0            | imm4          | 0   0
# | 0   1 | rw            | r0            | imm4          | 0   0

# ROR by immediate - Treat registers as 32 bit lanes and shift by immediate (
# | 1   0 | rw            | r0            | imm4          | 0   0
# | 1   1 | rw            | r0            | imm4          | 0   0

# 3 Reg operation - R1 is in the same 4 reg group as R0
# | LO    | rw            | r0            | r1    | 0   0 | 0   1
# | SO    | rw            | r0            | r1    | 0   1 | 0   1
# | AO    | rw            | r0            | r1    | 1   0 | 0   1
# | CO    | rw            | r0            | r1    | 1   1 | 0   1

# Wide arithmatic operation - operate on 32 bits
# 





# Logical Ops (LO)
EX_OP_AND       = 0b00
EX_OP_OR        = 0b01
EX_OP_XOR       = 0b10
EX_OP_NOT       = 0b11

# Shift Ops (SO)
EX_OP_ROR       = 0b00
EX_OP_SLA       = 0b01
EX_OP_SRA       = 0b10
EX_OP_SRL       = 0b11 # Right shift shifting in high bit (arithmatic shift)

# Arith Ops (AO)
EX_OP_ADD       = 0b00
EX_OP_SUB       = 0b01
EX_OP_MUL       = 0b10
EX_OP_MAC       = 0b11

# Crypto Ops (CO)
EX_OP_GF2       = 0b00
EX_OP_BITSLICE  = 0b01
EX_OP_PERMUTE   = 0b10
EX_OP_SBOX      = 0b11

# LoadImm/
EX_OP_MASK      = 0b00
EX_OP_LOAD8     = 0b01
EX_OP_XORC      = 0b10
EX_OP_RESERVED  = 0b11

# Size Tags
EX_SIZE_8       = 0b00
EX_SIZE_16      = 0b01
EX_SIZE_32      = 0b10

# Input/Output
INSTR_PULL_MSG  = 0b0000
INSTR_PULL_KEY  = 0b0001
INSTR_PUSH_OUT  = 0b0010

# Looping
INSTR_LOOP_BEGIN = 0b
INSTR_LOOP_END

#
HALT

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

def decode(instr, regfile):
    # Check if it's an R16-type instruction
    assert (instr & 0xF000) >> 12 != 0xF, "can only decode R-type instructions"

    # Simple decode
    op = (instr & 0xF000) >> 12
    rw = (instr & 0x000F) >> 0
    r1 = (instr & 0x0F00) >> 8
    r0 = (instr & 0x00F0) >> 4

    # Read from regfile if it's a non-load instruction
    v0 = self.regfile[r0]
    v1 = self.regfile[r1]

    # Write is always enabled for now
    write_en = 1

    return op, rw, write_en, v1, v0

def execute(op, v1, v0, acc):
    vw = np.zeros(shape=acc.shape, dtype=acc.dtype)
    # Bitwise operations
    if op == EX_OP_AND:
        vw = np.bitwise_and(v1, v0)
    elif op == EX_OP_OR:
        vw = np.bitwise_or(v1, v0)
    elif op == EX_OP_XOR:
        vw = v1 ^ v0
    elif op == EX_OP_NOT:
        vw = ~v1

    # Shift Operations
    elif op == EX_OP_ROR:
        vw = (v1 >> v0) | (v1 << 8 - v0)
    elif op == EX_OP_SLA:
        vw = (v1 << v0)
    elif op == EX_OP_SRA:
        vw = (v1 >> v0)
    elif op == EX_OP_SRL:
        vw = (v1 >> v0) # TODO - make arithmatic

    # Arithmatic
    elif op == EX_OP_ADD:
        vw = v1 + v0
    elif op == EX_OP_SUB:
        vw = v1 - v0
    elif op == EX_OP_MUL:
        vw = v1 * v0

    # Crypto ops
    elif op == EX_OP_BITSLICE:
        # Seperate v1 into 4 groups of 8 bits
        for i in range(4):
            for j in range(8):
                # We want to collect the kth bit of each element in v1[i*8 + j]
                # Create a mask we can apply to each byte
                mask = 0x01 << j
                for k in range(8):
                    vw[i*8 + j] |= ((v1[i*8 + k] & mask) >> i) << k

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
        vw[v0] = v1
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
        op, rw, write_en, v1, v0 = decode(instr, self.regfile)

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

def main(args=None):
    # Simple proxies to make our code prettier
    # The "Program" class only need integers
    R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15 = np.arange(16)

    # Program Definition
    p = Program()
    p.xor(R0, R0, R0)
    p.add8(R0, R0, R0)
    p.mul8(R1, R1, R0)

    # Interpreter
    i = Interpreter(program=p, regfile={i: i for i in range(16)})
    i.step()
    i.step()

if __name__ == '__main__':
    main()
