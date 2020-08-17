"""
Fuse

fuse用于融合两个iter，将两层循环合并到一层，其返回值为iter类型，可以多次合并。

"""

import tvm
from tvm import te
import numpy as np

n = te.var('n')
m = te.var('m')

A = te.placeholder((m, n), name='A')
B = te.compute((m, n), lambda i, j: A[i, j], name='B')

s = te.create_schedule(B.op)
# tile to four axises first: (i.outer, j.outer, i.inner, j.inner)
xo, yo, xi, yi = s[B].tile(B.op.axis[0], B.op.axis[1], x_factor=10, y_factor=5)
# then fuse (i.inner, j.inner) into one axis: (i.inner.j.inner.fused)
fused = s[B].fuse(xi, yi)
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
      for (i.inner.j.inner.fused: int32, 0, 50) {
        if @likely((((i.outer*10) + floordiv(i.inner.j.inner.fused, 5)) < m), dtype=bool, type="pure_intrin", index=0) {
          if @likely((((j.outer*5) + floormod(i.inner.j.inner.fused, 5)) < n), dtype=bool, type="pure_intrin", index=0) {
            B_2[((((i.outer*10) + floordiv(i.inner.j.inner.fused, 5))*stride_2) + (((j.outer*5) + floormod(i.inner.j.inner.fused, 5))*stride_3))] = (float32*)A_2[((((i.outer*10) + floordiv(i.inner.j.inner.fused, 5))*stride) + (((j.outer*5) + floormod(i.inner.j.inner.fused, 5))*stride_1))])
          }
        }
      }
    }
  }
}

// meta data omitted. you can use show_meta_data=True to include meta data

"""
