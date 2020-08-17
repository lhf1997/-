"""

Compute_inline

compute_inline把独立的计算操作转化成内联函数形式，在使用到原计算结果时再调用内联函数完成运算，通过compute_inline来减少一个stage。
can mark one stage as inline, then the body of computation will be expanded and inserted at the address where the tensor is required.

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
s[B].compute_inline()
print(tvm.lower(s, [A, B, C], simple_mode=True))


"""

Results:

primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {C: Buffer(C_2: handle, float32, [m: int32], [stride: int32], type="auto"),
             B: Buffer(B_2: handle, float32, [m], [stride_1: int32], type="auto"),
             A: Buffer(A_2: handle, float32, [m], [stride_2: int32], type="auto")}
  buffer_map = {C_1: C, A_1: A, B_1: B} {
  for (i: int32, 0, m) {
    C_2[(i*stride)] = (((float32*)A_2[(i*stride_2)]) + 1f32)*2f32)
  }
}

// meta data omitted. you can use show_meta_data=True to include meta data

"""
