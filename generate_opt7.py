import numpy as np

debug = False

"""
Each thread holds 32 * 4 registers
32 : each value corresponds to a value along the y-direction
4 :  each value corresponds to a value along the z-direction, but strided by the number of
warps (8)
e.g., the first thread in the first warp is responsible for storing A[0,:,0,8,16,24] in
registers.
"""

def declare():
    print("#define DECLARE_REGISTERS ", end=" ")
    for k in range(4):
        for j in range(32):
            print("float p_%d_%d = 0.f;" % (k, j), end=" ")
    print()

def load_shared():
    print("#define LOAD_SHARED(__PLANE__) {", end=" ")
    print("int idx_fixed = idx + snxy * idy;", end=" ")
    for plane in range(4):
        print("if ( %d == (__PLANE__)) { " % plane, end=" ")
        for j in range(32):
            print("p_%d_%d = smem[idx_fixed + snx * %d];" % (plane, j, j), end=" ")
        print("}", end=" ")
    print("}")

def store_shared():
    print("#define STORE_SHARED(__BATCH_Y__) {", end=" ")
    print("int idx_fixed = idx + snx * idy; int sptr;", end=" ")
    for batch_y in range(4):
        print("if ( %d == (__BATCH_Y__)) { " % batch_y, end=" ")
        for y in range(8):
            for plane in range(4):
                print("sptr = idx_fixed + snx * 8 * %d + snxy * %d;" % (plane, y), end=" ")
                print("smem[sptr] = p_%d_%d;" % (plane, y + 8 * batch_y), end=" ")
        print("}", end=" ")
    print("}")


declare()
load_shared()
store_shared()

