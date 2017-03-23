#Instruction Set simulator for CESEL#

Author: Kevin Kiningham

## How to use ##

To create a new program:

```python
p = cesel.Program('Program Name')
p.add(1, 2, 3) # Add register 2 to register 3 and store in register 1
# etc
```

To simulate a program:

```python
# Create a new Interpreter with the register file initialized to all ones
s = cesel.Interpreter(program, regfile=1)

# Step the simulation
s.step()

# See the value of the first register
print s.regfile[0]
```
