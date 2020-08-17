"""
Reorder

reorder用于重置循环iter的内外顺序，根据局部性原理，最大化利用cache中的现有数据，减少反复载入载出的情况。
注意，这里到底怎样的顺序是最优化的是一个很有趣的问题。以矩阵乘法为例，M, N, K三维，往往是将K放在最外层可以最大程度利用局部性。这个具体例子，具体探究。

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
# then reorder the axises: (i.inner, j.outer, i.outer, j.inner)
s[B].reorder(xi, yo, xo, yi)
print(tvm.lower(s, [A, B], simple_mode=True))


"""

Results:

primfn(A_1: handle, B_1: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_2: handle, float32, [m: int32, n: int32], [stride: int32, stride_1: int32], type="auto"),
             B: Buffer(B_2: handle, float32, [m, n], [stride_2: int32, stride_3: int32], type="auto")}
  buffer_map = {B_1: B, A_1: A} {
  for (i.inner: int32, 0, 10) {
    for (j.outer: int32, 0, floordiv((n + 4), 5)) {
      for (i.outer: int32, 0, floordiv((m + 9), 10)) {
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
