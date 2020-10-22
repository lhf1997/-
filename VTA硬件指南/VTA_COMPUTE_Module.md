# VTA_COMPUTE_Module

本文主要介绍VTA的COMPUTE模块，结合hls源码，分析GEMM和ALU指令的具体用法。



[TOC]

## Architecture

COMPUTE模块如下所示，由Reg File、uop cache、GEMM核和ALU构成。与LOAD模块通过LOAD BUFFER、UOP SRAM、ACTIVATION SRAM和KERNEL SRAM进行数据传输和共享，与STORE模块通过STORE BUFFER进行交互。COMPUTE Q用于接收FETCH模块发送过来的GEMM或ALU指令。

<img src="C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20201013193422875.png" alt="image-20201013193422875" style="zoom: 50%;" />



## VTA’s On-Chip SRAMs

VTA具有三种不同的存储范围，每个范围对应于不同的片上SRAM缓冲区。

​	inp_mem：input buffer，默认存储数据类型为int8，数据形状为（1,16）大小的矩阵，即（BATCH，BLOCK_IN），该SRAM大小为32KB，以上数据用户均可通过配置`vta_config.json`自定义。

​	wgt_mem：weight buffer，默认存储数据类型为int8，数据形状为（16,16）大小的矩阵，即（BLOCK_OUT，BLOCK_IN），该SRAM大小为256KB。

​	acc_mem：acc buffer，默认存储数据类型为int32，数据形状为（1,16）大小的矩阵，即（BATCH，BATCH_OUT），大小为128KB。该mem既包含卷积和矩阵乘法中的中间结果，又包含池化、batch normal和激活层的中间结果。



## GEMM Core

VTA中的COMPUTE模块在整个设计中充当RISC处理器的角色，主要进行张量级别的操作，主要包括两个功能单元：GEMM和ALU。GEMM主要进行乘加运算，实现神经网络模型中的卷积层和全连接层，GEMM核中最小的计算单元如下图所示，该矩阵乘法为`（BATCH，BLOCK_IN）×（BLOCK_OUT， BLOCK_IN）`，数据分别存储在inp_mem和wgt_mem中，最后将计算的结果存储在acc_mem中等待累加。

<img src="C:\Users\刘鸿飞\Desktop\vta开发文档\picture\tensor_core.png" style="zoom: 33%;" />

每一条GEMM指令会对应到一组micro-op，如下图所示，uop存放在uop buffer中，包含acc_idx、inp_idx和wgt_idx这三个字段，提供数据的索引地址。GEMM指令根据uop提供的索引将计算过程循环嵌套成多个for循环，以保证GEMM核在每个周期可以执行一次`（BATCH，BLOCK_IN）×（BLOCK_OUT， BLOCK_IN）`。如下面伪代码所示，最内层循环对inp和wgt数据进行矩阵乘法，将结果累加到acc_mem中，同时根据GEMM指令中的字段更新下次循环所需数据存放在buffer中的索引。![](C:\Users\刘鸿飞\Desktop\vta开发文档\picture\gemm_core.png)

## Forward analysis	

以2维卷积为案例，正向分析其完整的工作流程。本次分析的卷积运算如下所示，这里将硬件参数设置为：

```shell
# 其他参数默认，未作修改
VTA_BATCH = 1
VTA_BLOCK_OUT = 32
VTA_BLOCK_IN = 32
```

​	卷积运算的具体参数如下，不再赘述。

```shell
Conv2DWorkload(batch=1, height=8, width=8, in_filter=32, out_filter=32,hkernel=3, wkernel=3, hpad=1, wpad=1, hstride=1, wstride=1) 
```

通常，一个完整的卷积过程会如下图所示，主要包括DMA Load、Conv2D、Shr、Max、Min和DMA Store这几个部分。

![image-20201020144137235](C:\Users\刘鸿飞\AppData\Roaming\Typora\typora-user-images\image-20201020144137235.png)

当如上的卷积运算任务准备就绪后，VTA首先会对数据的格式分析，先将其转化为NCHW的格式。随后，VTA会再将数据进行打包，将数据格式转换为NCHWnc，对于本例来说，打包后的数据格式如下：

```python
# Input feature map: (N, IC, H, W, n, ic)
data_shape = (1, 1, 8，8，1, 32)

# Kernel: (OC, IC, H, W, oc, ic)
kernel_shape = (1, 1, 8，8，32, 32)

# Output feature map: (N, OC, H, W, n, oc)
output_shape = (1, 1, 8, 8, 1, 32)
```

由于在N和C的维度上均为1，此卷积计算不需要进行分块，可以一次从DRAM加载到VTA的片上buffer中。且只有一次的计算任务，VTA内部没有调用Virtual Threading以隐藏内存访问的延时。

VTA会将计算的任务详细打包成一个schedule（包括数据类型定义、内存访问过程、循环迭代过程等），软件栈会将这样一个schedule进行解析，解释成一条条指令的层面，随后主机会将这些指令以及VTA runtime从DRAM发送到VTA上，以pynq板卡为例，主机和FPGA之间的数据通信是通过RPC远程过程调用来实现的。

卷积计算的一系列指令首先会被发送到FETCH模块，FETCH会将这些指令分别发送到LOAD、COMPUTE和STORE模块中，由于只执行一次卷积计算，所以模块之间的调度比较简单，可以简单理解为LOAD、COMPUTE和STORE之间串行执行。完整的指令流程：



Load Uop

首先，LOAD指令会从DRAM中加载地址数据到uop_mem中，该过程提供需要计算数据的索引。每一组uop指令会对应到一条GEMM或ALU指令的操作，instruction：0作为第一条指令，会先对gemm指令进行初始化操作，uop字段中的索引均为0，即`[0000] acc=0, inp=0, wgt=0`  。该LOAD指令不进行其他操作，所以四位DEPT FLAGS均为零，DRAM和SRAM地址均为初始地址。

​	

GEMM Reset

uop加载后，将地址索引推入compute模块，开始执行gemm指令，由于uop提供的索引均为零，此时gemm指令不进行任何计算任务，DEPT FLAGS会将push prev写为1，同时将g2l_dep_queue也写为1，对应hls源码如下，表示compute中有指令在进行。

```c++
  // Push dependence token if instructed
  if (insn.generic.push_prev_dep) {
    g2l_dep_queue.write(1);
  }
```

该过程的DEPT FLAGS为：`dep - pop prev: 0, pop next: 0, push prev: 1, push next: 0` ， 是GEMM指令中自带的四位指令字段。该过程的queue消息为：`l2g_queue = 0， g2l_queue = 1，s2g_queue = 0, g2s_queue = 0`   ，是VTA用于检测模块计算进度的标识，不在指令字段中，用于解决数据相关性的问题。为了便于理解，GEMM Reset的伪代码相应可以表示为如下。

```python
for b in range(0, batch):
    for co in range(0, channel_out):
        for h in range(0, height):
            for w in range(0, width):
                output[b][co][h][w] = 0
```



Load Input

该LOAD指令会根据dram_base字段的地址加载存放在DRAM中的input数据，再根据sram_base字段的地址放入片上inp_mem中。对应为`DRAM: 0x00000100, SRAM:0x0000`。该卷积计算任务的fmap的h=8，w=8，VTA会将该数据padding后加载到片上buffer对应为：。DEPT FLAGS为0100（指令中已指定的字段），会将pop next 写为1，表示通知compute模块有数据要load进来。

```c++
  // Pop dependence token if instructed
  if (insn.pop_next_dep) {
    g2l_dep_queue.read();
  }
```

该过程的DEPT FLAGS为：`dep - pop prev: 0, pop next: 1, push prev: 0, push next: 0`  ，上述代码段表示检测compute模块是否运算结束，该过程的queue消息为：`l2g_queue = 0, g2l_queue = 0, s2g_queue = 0, g2s_queue = 0`。由于与gemm的reset不存在数据相关性，g2l_queue在这里被清空，相应的软件代码如下所示。

```c++
 else if (c.mem.opcode == VTA_OPCODE_LOAD &&
        (c.mem.memory_type == VTA_MEM_ID_INP || c.mem.memory_type == VTA_MEM_ID_WGT)) {
        if (c.mem.pop_next_dep) g2l_queue--; // g2l_queue在这里被清空
        if (c.mem.push_next_dep) l2g_queue++;
      }
```



Load Weight

该LOAD指令会根据dram_base字段的地址加载存放在DRAM中的weight数据，再根据sram_base字段的地址放入片上wgt_mem中。卷积核大小为3×3，对应到内存中是从0-8共九位连续存放的字段。加载Weight时的DEPT FLAGS为0001，将push next写为1，物理含义即为将加载的数据通过Load与Compute模块之间的Buffer将load好的数据传递进去。

```c++
  // Push dependence token if instructed
  if (insn.push_next_dep) {
    l2g_dep_queue.write(1);
  }
}
```

该过程的DEPT FLAGS为：`dep - pop prev: 0, pop next: 0, push prev: 0, push next: 1`，相应的queue为：`l2g_queue = 1, g2l_queue = 0,s2g_queue = 0, g2s_queue = 0`。



Load Uop

在数据加载完之后，VTA会加载uop指令给gemm指令提供计算地址的索引，gemm指令的最内层循环没进行一次矩阵乘法运算，uop就要给该运算提供相应的inp、wgt和acc buffer的索引一次，VTA中的auto tunning会找到卷积分块的最优解，本次卷积计算任务不涉及分块。

```c++
for (i = 0; i < iter_out; i++) {
	for (j = 0; j < iter_in; j++) {
      	for (k = uop_bgn; k < uop_end; k++) {
       		// Read micro op
       		uop_T uop = uop_mem[k];
       		// Read in memory indices
       		acc_idx_T acc_idx = uop.dst_idx;
       		inp_idx_T inp_idx = uop.inp_idx;
       		wgt_idx_T wgt_idx = uop.wgt_idx;
       		// Update those indices with the following affine functions
       		acc_idx += iter_in * dst_factor_in + iter_out * dst_factor_out;
       		inp_idx += iter_in * src_factor_in + iter_out * src_factor_out;
       		wgt_idx += iter_in * wgt_factor_in + iter_out * wgt_factor_out;
       		// Perform GEMM operation
       		acc_mem[acc_idx] += dot(inp_mem[inp_idx], wgt[wgt_idx]);
       	}
    }
}
```

该过程的DEPT FLAGS为：`dep - pop prev: 1, pop next: 0, push prev: 0, push next: 0`  相应的queue为：`l2g_queue = 0, g2l_queue = 0, s2g_queue = 0, g2s_queue = 0`  。



GEMM

GEMM指令执行矩阵的乘加运算，实际上，VTA在针对某一具体的卷积计算任务时，会将该具体的卷积计算任务分解成一层一层循环进行计算。

