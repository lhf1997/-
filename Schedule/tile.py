import tvm
from tvm import te
import numpy as np

# declare some variables for use later
n = te.var('n')
m = te.var('m')

A = te.placeholder((m, n), name='A')
B = te.compute((m, n), lambda i, j: A[i, j], name='B')

s = te.create_schedule(B.op)
xo, yo, xi, yi = s[B].tile(B.op.axis[0], B.op.axis[1], x_factor=10, y_factor=5)
print(tvm.lower(s, [A, B], simple_mode=True))


"""

Results:

primfn(A_1: handle, B_1: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_2: handle, float32, [m: int32, n: int32], [stride: int32, stride_1: int32], type="auto"),
             B: Buffer(B_2: handle, float32, [m, n], [stride_2: int32, stride_3: int32], type="auto")}
  buffer_map = {B_1: B, A_1: A} {
  for (i.outer: int32, 0, floordiv((m + 9), 10)) {
    for (j.outer: int32, 0, floordiv((n + 4), 5)) {
      for (i.inner: int32, 0, 10) {
        for (j.inner: int32, 0, 5) {
          if @likely((((i.outer*10) + i.inner) < m), dtype=bool, type="pure_intrin", index=0) {
            if @likely((((j.outer*5) + j.inner) < n), dtype=bool, type="pure_intrin", index=0) {
              B_2[((((i.outer*10) + i.inner)*stride_2) + (((j.outer*5) + j.inner)*stride_3))] = (float32*)A_2[((((i.outer*10) + i.inner)*stride) + (((j.outer*5) + j.inner)*stride_1))])
            }
          }
        }
      }
    }
  }
}

// meta data omitted. you can use show_meta_data=True to include meta data

"""
