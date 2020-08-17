"""
Bind

bind可以将指定的轴与线程轴绑定，通常在gpu编程中使用。

"""

import tvm
from tvm import te
import numpy as np

n = te.var('n')
m = te.var('m')

A = te.placeholder((n,), name='A')
B = te.compute(A.shape, lambda i: A[i] * 2, name='B')

s = te.create_schedule(B.op)
bx, tx = s[B].split(B.op.axis[0], factor=64)
s[B].bind(bx, te.thread_axis("blockIdx.x"))
s[B].bind(tx, te.thread_axis("threadIdx.x"))
print(tvm.lower(s, [A, B], simple_mode=True))


"""

Results:

primfn(A_1: handle, B_1: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_2: handle, float32, [n: int32], [stride: int32], type="auto"),
             B: Buffer(B_2: handle, float32, [n], [stride_1: int32], type="auto")}
  buffer_map = {B_1: B, A_1: A} {
  attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = floordiv((n + 63), 64);
  attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 64;
  if @likely((((blockIdx.x*64) + threadIdx.x) < n), dtype=bool, type="pure_intrin", index=0) {
    B_2[(((blockIdx.x*64) + threadIdx.x)*stride_1)] = ((float32*)A_2[(((blockIdx.x*64) + threadIdx.x)*stride)])*2f32)
  }
}

// meta data omitted. you can use show_meta_data=True to include meta data


"""
