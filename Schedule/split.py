"""

Split

split是fuse的反操作，把iter以factor为间隔分离成outer与inner两层迭代，增加循环层数，用于将循环操作分割为更小的子任务。
事实上，以CUDA为例，gridDim和blockDim都可以最多是三维，所以通过split可以产生新的维度用于绑定到grid和block上

"""

import tvm
from tvm import te
import numpy as np

# declare some variables for use later
n = te.var('n')
m = te.var('m')

A = te.placeholder((m,), name='A')
B = te.compute((m,), lambda i: A[i]*2, name='B')

s = te.create_schedule(B.op)

xo, xi = s[B].split(B.op.axis[0], factor=32) # split can split a specified axis into two axises by factor.
print(tvm.lower(s, [A, B], simple_mode=True))


A = te.placeholder((m,), name='A')
B = te.compute((m,), lambda i: A[i], name='B')

s = te.create_schedule(B.op)
bx, tx = s[B].split(B.op.axis[0], nparts=32) #  split a axis by nparts, which splits the axis contrary with factor
print(tvm.lower(s, [A, B], simple_mode=True))

"""
Results:

primfn(A_1: handle, B_1: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_2: handle, float32, [m: int32], [stride: int32], type="auto"),
             B: Buffer(B_2: handle, float32, [m], [stride_1: int32], type="auto")}
  buffer_map = {B_1: B, A_1: A} {
  for (i.outer: int32, 0, floordiv((m + 31), 32)) {
    for (i.inner: int32, 0, 32) {
      if @likely((((i.outer*32) + i.inner) < m), dtype=bool, type="pure_intrin", index=0) {
        B_2[(((i.outer*32) + i.inner)*stride_1)] = ((float32*)A_2[(((i.outer*32) + i.inner)*stride)])*2f32)
      }
    }
  }
}

// meta data omitted. you can use show_meta_data=True to include meta data
primfn(A_1: handle, B_1: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_2: handle, float32, [m: int32], [stride: int32], type="auto"),
             B: Buffer(B_2: handle, float32, [m], [stride_1: int32], type="auto")}
  buffer_map = {B_1: B, A_1: A} {
  for (i.outer: int32, 0, 32) {
    for (i.inner: int32, 0, floordiv((m + 31), 32)) {
      if @likely(((i.inner + (i.outer*floordiv((m + 31), 32))) < m), dtype=bool, type="pure_intrin", index=0) {
        B_2[((i.inner + (i.outer*floordiv((m + 31), 32)))*stride_1)] = (float32*)A_2[((i.inner + (i.outer*floordiv((m + 31), 32)))*stride)])
      }
    }
  }
}

// meta data omitted. you can use show_meta_data=True to include meta data

"""
