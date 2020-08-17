"""

Compute_root

compute_root是compute_at的反操作。因为不做任何schedule的话，每一个stage默认就是compute_root的，这个schedule相当于注释了对之前对一个stage的compute操作。
can move computation of one stage to the root.

"""

import tvm
from tvm import te
import numpy as np

n = te.var('n')
m = te.var('m')

A = te.placeholder((m,), name='A')
B = te.compute((m,), lambda i: A[i]+1, name='B')
C = te.compute((m,), lambda i: B[i]*2, name='C')

s = te.create_schedule(C.op)
s[B].compute_at(s[C], C.op.axis[0])
s[B].compute_root()
print(tvm.lower(s, [A, B, C], simple_mode=True))


"""

Results:

primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {B: Buffer(B_2: handle, float32, [m: int32], [stride: int32], type="auto"),
             C: Buffer(C_2: handle, float32, [m], [stride_1: int32], type="auto"),
             A: Buffer(A_2: handle, float32, [m], [stride_2: int32], type="auto")}
  buffer_map = {C_1: C, A_1: A, B_1: B} {
  for (i: int32, 0, m) {
    B_2[(i*stride)] = ((float32*)A_2[(i*stride_2)]) + 1f32)
  }
  for (i_1: int32, 0, m) {
    C_2[(i_1*stride_1)] = ((float32*)B_2[(i_1*stride)])*2f32)
  }
}

// meta data omitted. you can use show_meta_data=True to include meta data

"""
