"""
schedule的使用

什么是schedule？
通常存在几种计算相同结果的方法，但是，不同的方法将导致不同的位置和性能。因此TVM要求用户提供如何执行称为Schedule的计算。

Schedule Primitives in TVM
https://tvm.apache.org/docs/tutorials/language/schedule_primitives.html
"""

from tvm import te
import tvm

# declare some variables for use later
n = te.var('n') # Create a new variable with specified name and dtype
m = te.var('m')

# declare a matrix element-wise multiply 
A = te.placeholder((m, n), name='A') # te.placeholder: Construct an empty tensor object. 
B = te.placeholder((m, n), name='B')
C = te.compute((m, n), lambda i, j: A[i, j] * B[i, j], name='C') # te.compute: Construct a new tensor by computing over the shape domain. The compute rule is result[axis] = fcompute(axis)

# Create a schedule for list of ops
s = te.create_schedule([C.op]) 
# lower will transform the computation from definition to the real callable function. With argument `simple_mode=True`, it will
# return you a readable C like statement, we use it here to print the schedule result.
print(tvm.lower(s, [A, B, C], simple_mode=True))

# te.lower
# Lowering step before build into target.

"""
Results:
primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {C: Buffer(C_2: handle, float32, [m: int32, n: int32], [stride: int32, stride_1: int32], type="auto"),
             B: Buffer(B_2: handle, float32, [m, n], [stride_2: int32, stride_3: int32], type="auto"),
             A: Buffer(A_2: handle, float32, [m, n], [stride_4: int32, stride_5: int32], type="auto")}
  buffer_map = {C_1: C, A_1: A, B_1: B} {
  for (i: int32, 0, m) {
    for (j: int32, 0, n) {
      C_2[((i*stride) + (j*stride_1))] = ((float32*)A_2[((i*stride_4) + (j*stride_5))])*(float32*)B_2[((i*stride_2) + (j*stride_3))]))
    }
  }
}
// meta data omitted. you can use show_meta_data=True to include meta data
"""

# Stage
# One schedule is composed by multiple stages, and one Stage represents schedule for one operation. We provide various methods to schedule every stage.
