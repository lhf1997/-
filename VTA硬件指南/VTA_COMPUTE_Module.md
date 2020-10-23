# VTA_COMPUTE_Module

本文主要介绍VTA的COMPUTE模块，结合hls源码，分析GEMM和ALU指令的具体用法。



[TOC]

## Architecture

COMPUTE模块如下所示，由Reg File、uop cache、GEMM核和Tensor ALU构成。COMPUTE模块与LOAD模块通过INPUT BUFFER、WEIGHT BUFFERUOP SRAM进行数据传输和共享，与STORE模块通过OUTPUT BUFFER进行交互。COMPUTE CMD Q用于接收FETCH模块发送过来的GEMM或ALU指令。

<img src="C:\Users\pc\Desktop\vta\图片\vta_overview.png" style="zoom: 20%;" />



## VTA’s On-Chip SRAMs

VTA具有三种不同的存储范围，每个范围对应于不同的片上SRAM缓冲区。

​	inp_mem：input buffer，默认存储数据类型为int8，该SRAM大小为32KB，SRAM大小可通过配置`vta_config.json`自定义。

​	wgt_mem：weight buffer，默认存储数据类型为int8，该SRAM大小为256KB。

​	acc_mem：acc buffer，默认存储数据类型为int32，大小为128KB。该mem既包含卷积和矩阵乘法中的中间结果，又包含非线性函数计算的中间结果。



## GEMM Core

VTA中的COMPUTE模块在整个设计中主要进行张量级别的操作，主要包括两个功能单元：GEMM和ALU。GEMM主要进行乘加运算，实现神经网络模型中的卷积层和全连接层，GEMM核中最小的计算单元如下图所示，该矩阵乘法为`（BATCH，BLOCK_IN）×（BLOCK_OUT， BLOCK_IN）`，数据分别存储在inp_mem和wgt_mem中，最后将计算的结果存储在acc_mem中等待累加。

<img src="C:\Users\pc\Desktop\vta\图片\tensor_core.png" style="zoom: 33%;" />

每一条GEMM指令会对应到一组micro-op，如下图所示，uop存放在uop buffer中，包含acc_idx、inp_idx和wgt_idx这三个字段，提供数据的索引地址。GEMM指令根据uop提供的索引将计算过程循环嵌套成多个for循环，以保证GEMM核在每个周期可以执行一次`（BATCH，BLOCK_IN）×（BLOCK_OUT， BLOCK_IN）`。如下面伪代码所示，最内层循环对inp和wgt数据进行矩阵乘法，将结果累加到acc_mem中，同时根据GEMM指令中的字段更新下次循环所需数据存放在buffer中的索引。

<img src="C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20201022191246618.png" alt="image-20201022191246618" style="zoom: 50%;" />



## DataFlow

下图所示为VTA的数据流，假设这里的MODULE是COMPUTE Module，由于LOAD、COMPUTE、STORE三个模块存在共用buffer的情况，所以不可避免地会出现数据相关性的问题。VTA为了解决这个问题，在LOAD、GEMM、ALU、STORE指令中引入了四位的DEPT FLAGS，分别表示为：`pop prev, pop next, push prev, push next`。 



<img src="C:\Users\pc\Desktop\vta\图片\dataflow.png" style="zoom:50%;" />

结合上图右侧的伪代码，且对应到上图四位DEPT FLAGS分别为：

```powershell
pop prev # producer RAW queue， 向前一queue读取FIFO
pop next # consumer RAW queue， 向后一queue读取FIFO
push prev # producer WAR queue, 向前一queue的FIFO写1
push next # consumer WAR queue， 向后一queue的FIFO写1
```



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

<img src="C:\Users\pc\Desktop\vta\图片\conv2d_dataflow.png" style="zoom:25%;" />

当如上的卷积运算任务准备就绪后，VTA会将数据进行打包，改变相邻通道上的数据在内存中的存储结构，将数据格式从NCHW转换为NCHWnc，对于本例来说，打包后的数据格式如下：

```python
# Input feature map: (N, IC, H, W, n, ic)
data_shape = (1, 1, 8，8，1, 32)

# Kernel: (OC, IC, H, W, oc, ic)
kernel_shape = (1, 1, 8，8，32, 32)

# Output feature map: (N, OC, H, W, n, oc)
output_shape = (1, 1, 8, 8, 1, 32)
```

VTA中使用auto tuning可以结合目标FPGA平台的计算性能和带宽，对卷积计算任务在输入和输出通道上展开，这里手动设置参数不进行分块，假设将INP、WGT一次从DRAM加载到VTA的片上buffer中。且由于只有一次的计算任务，VTA内部没有启动Virtual Threading以隐藏内存访问的延时。

VTA会将计算的任务详细打包成一个schedule（包括数据类型定义、内存访问过程、循环迭代过程等），软件栈会将这样一个schedule进行解析，打包成部分指令和API结合的模式，如下图所示。随后主机会将这些API、bit流文件以及VTA runtime从DRAM发送到VTA上，以pynq板卡为例，主机和FPGA之间的数据通信是通过RPC远程过程调用来实现的。

ARM核会向VTA发送启动信号以开启PL侧的VTA，FEICH模块会从DRAM中以DMA的方式读取已准备好的指令流，访问指令流后，FETCH模块会对其进行译码，将这些指令分别发送到LOAD、COMPUTE和STORE的命令队列中。由于只执行一次卷积计算，所以模块之间的调度比较简单，可以简单理解为LOAD、COMPUTE和STORE之间串行执行。完整的指令流程如下：



Load Uop

首先，LOAD指令会从DRAM中加载地址数据到uop_mem中，该过程提供需要计算数据的索引。每一组uop指令会对应到一条GEMM或ALU指令的操作，首先对gemm指令进行初始化操作，uop字段中的索引均为0，即`[0000] acc=0, inp=0, wgt=0`  。该LOAD指令在COMPUTE模块中执行，所以pop prev为1，表示向LD→CMP Q读取FIFO。

![image-20201022202547884](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20201022202547884.png)

该指令为：

```shell
# 操作码
OPCODE：0					#4位
# 4为数据相关性标识
DEPT FLAGS：0010
# 片上缓存buffer对象
BUFFER_ID：uop_mem			#2位
# 片上缓存基地址
SRAM_BASE ：0x0000			#16位
# DRAM基地址
DRAM_BASE ：0x00001400		#32位
# 以下为DMA参数
Y SIZE: 0000000000000001	#16位 
X SIZE: 0000000000000001	#16位	
X STRIDE: 0000000000000001	#16位
Y_PAD_0: 0000				#4位
Y_PAD_1: 0000				#4位
X_PAD_0: 00					#2位
X_PAD_1: 00					#2位
```



GEMM Reset

uop加载后，将地址索引推入compute模块，开始执行gemm指令，由于uop提供的索引均为零，此时gemm指令不进行任何计算任务，该指令DEPT FLAGS字段的push prev为1，该指令在COMPUTE模块中执行，表示向CMP→LD Q中写1。

![image-20201022210852310](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20201022210852310.png)

该指令为：

```shell
OPCODE：2
DEPT FLAGS：0010
RESET:1						#RESET		
MICRO-OP BEGIN: 0			#uop起始地址
MICRO-OP END: 1				#uop结束地址，只用到一条
LOOP EXTENT 0: 8			#Irerations in the outer uop exe loop，与fmap大小有关
LOOP EXTENT 1: 8			#...inner uop ... ，与famp有关
# 以下为DMA参数
ACCUM IDX FACTOR 0: 1		#acc_mem外层循环地址索引
ACCUM IDX FACTOR 1:	8		#acc_mem内存循环地址索引，将会用到的acc_mem清空
INPUT IDX FACTOR 0:	0 		#inp_mem外层循环地址索引
INPUT IDX FACTOR 1:	0 		#inp_mem内存循环地址索引
WEIGHT IDX FACTOR 0: 0		#wgt_mem外层循环地址索引
WEIGHT IDX FACTOR 1: 0		#wgt_mem内存循环地址索引
```



Load Input

具体指令字段见上图，该卷积计算任务的fmap的h=8，w=8，DEPT FLAGS为0100，即pop next 为1，表示读取LD→CMP Q。

```shell
OPCODE：0					
DEPT FLAGS：0010
BUFFER_ID：inp_mem			
SRAM_BASE ：0x0000			
DRAM_BASE ：0x000000100		
# 以下为DMA参数
Y SIZE: 0000000000001000	#8, fmap为8×8
X SIZE: 0000000000001000	#8	
X STRIDE: 0000000000001000	#8
Y_PAD_0: 0001				#1， 卷积运算padding=1
Y_PAD_1: 0001				#1
X_PAD_0: 01					#1
X_PAD_1: 01					#1
```

本次卷积计算中，加载input的伪代码可以表示为：

```python
for H in range(0,10)
	for W in range(0,10)
    	for c in range(0,16)
        	inp_mem[N][C][H][W] = DRAM_INP[N][C][H][W]
```



Load Weight

该卷积核大小为3×3，DEPT FLAGS为0001，即push next为1，表示向CMP→ST Q中写1。

```shell
OPCODE：0					
DEPT FLAGS：0001
BUFFER_ID：wgt_mem			
SRAM_BASE ：0x0000			
DRAM_BASE ：0x000000020		
# 以下为DMA参数
Y SIZE: 0000000000000001	#1
X SIZE: 0000000000001001	#9
X STRIDE: 0000000000001001	#9，3×3卷积核在内存中连续存放
Y_PAD_0: 0000				
Y_PAD_1: 0000				
X_PAD_0: 01					
X_PAD_1: 01				
```



Load Uop

经过以上几条指令，input fmap与weight分别从DRAM中存放到inp_mem和wgt_mem中，此时，需要加载uop指令，向gemm运算提供计算地址的索引，gemm指令的最内层循环每进行一次矩阵乘法运算，uop就要给该运算提供相应的inp、wgt和acc buffer的索引一次。该指令在COMPUTE模块中进行，DEPT FLAGS为1000，即pop prev=1，表示读LD→CMP的FIFO。acc、inp和mem的起始索引均为0，每次循环都会进行更新，如下代码块。本次计算外层循环为8，内存循环为3，每次会用到24条uop指令。

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

相应伪代码可以表示为：

```python
for i0 in range(0,8):
	for i1 in range(0,3):
		for uop_idx in range (1,25):
			x, y, z = decode_gemm_indices(uop_buffer[upc])
            reg_idx = i0 * x0 + i1 * x1 + x
            inp_idx = i0 * y0 + i1 * y1 + y
            wgt_idx = i0 * z0 + i1 * z1 + z
            reg_file[reg_idx] += GEMM(inp_buff[inp_idx], wgt_buff[wgt_idx])
```

其中，x0，x1，y0，y1，z0，z1分别表示GEMM指令中的部分字段。



GEMM

GEMM指令执行矩阵的乘加运算，实际上，VTA在针对某一具体的卷积计算任务时，会将该具体的卷积计算任务分解成一层一层循环进行计算。VTA中通过primitives会对该计算过程的细节进行调整，目的是在有限计算资源的情况下，优化计算资源的利用率。最常见的三种primitives有split（将一层循环，分成内外两层循环），reorder（对多层循环的循环顺序进行重新排序），compute_at（在指定的某层循环内融合两个循环）。基本的卷积过程如下，对于最内层循环，首先读取uop_mem中inp、wgt、acc地址的索引，随后将索引进行更新，并执行gemm操作，最后将结果写入acc_mem。对于标准卷积来说，可以表示为如下所示：

```python
# after reset operation
# conv2d
for ci in range(0, channel_in):
    for kh in range(0, kernel_height):
          for kw in range(0, kernel_weight):
                output[b][co][h][w] += weight[co][ci][kh][kw] * 
                            		   feature[b][ci][h*stride+kh][w*stride+kw]
output[b][co][h][w] += bias[b][co][h][w]
```

实际上，VTA会对卷积运算在输出通道，高和宽上进行分块（通过auto tuing找到最佳参数），将卷积的计算循环展开。随后会将卷积的循环进行重排，将块内循环放入循环内侧，如下所示。

```python
# 忽略load input，load weight
for b in range(0, batch):
    for h1 in range(0, height/tile_h):
        for co1 in range(0, channel_out/tile_co):
            for w1 in range(0, width/tile_w):
                #conv2d
                for co2 in range(0, tile_co):
                    for h2 in range(0, tile_h):
                        for w2 in range(0, tile_w):
                            output[b][co1][co2][h1][h2][w1][w2] = 0
                            for ci in range(0, channel_in):
                    			for kh in range(0, kernel_height):
                                    for kw in range(0, kernel_weight):
                                        output[b][co1][co2][h1][h2][w1][w2] += weight[co1][co2][ci][kh][kw]
* 																							
                                        feature[b][ci][(h1*tile_h+h2)*stride+kh][(w1*tile_w+w2)*stride+kw]
    
    						output[b][co1][co2][h1][h2][w1][w2] += bias[b][co1][co2][h1][h2][w1][w2]
```

之后VTA还会对其进行reorder，最大化利用cache中的现有数据，减少反复载入载出的情况。在这一步会往整体循环中插入LOAD/SOTRE指令，完成最终的调度。

```shell
OPCODE：0010
DEPT FLAGS：0010				#push prev=1，向前一个FIFO写1
RESET:0						
MICRO-OP BEGIN: 1			#接着第一条uop的地址
MICRO-OP END: 25			#共用到8*3=24条uop
LOOP EXTENT 0: 8			
LOOP EXTENT 1: 3			
ACCUM IDX FACTOR 0: 1		
ACCUM IDX FACTOR 1:	0		
INPUT IDX FACTOR 0:	1		
INPUT IDX FACTOR 1:	1 		
WEIGHT IDX FACTOR 0: 0		
WEIGHT IDX FACTOR 1: 1		
```



SHR、MIN、MAX

在VTA中gemm部分计算结束后，会对结果进行SHR、MIN和MAX操作，用于实现模拟Relu非线性函数的效果，具体算法不做深入。SHR、MIN和MAX三种操作也同样实在COMPUTE模块中通过ALU来实现的，与GEMM一样，每次进行运算之前都需要通过uop指令提供计算数据的地址索引。这里只对ALU SHR进行分析，其他不再赘述。

```shell
OPCODE：1000
DEPT FLAGS：0000
RESET:0						#RESET		
MICRO-OP BEGIN: 25			#
MICRO-OP END: 26			#只用到一条
LOOP EXTENT 0: 1			#
LOOP EXTENT 1: 64			#对8×8矩阵进行shr
# 以下为DMA参数
DST IDX FACTOR 0: 1			
DST IDX FACTOR 1: 0			
SRC IDX FACTOR 0: 1 		
SRC IDX FACTOR 1: 0 		
ALU OPCODE IDX : 0			#SHR、MIN或MAX的opcode
USE_IMM: 0
IMM: 0
```



STORE

STORE可以视为LOAD指令的反向操作，与LOAD指令使用的指令一段一致，在Relu进行完毕之后，数据将存放在acc_mem中，STORE就是将该数据以循环读取的方式写入DRAM中，通常在一个周期内是写不完的。DEPT FLAGS为1010，由于可能存在数据相关性，所以需要向CMP→ST FIFO中读，并向ST→CMP FIFO中写1。

```shell
OPCODE：1					
DEPT FLAGS：1010
BUFFER_ID：acc_mem			
SRAM_BASE ：0x0000			
DRAM_BASE ：0x000000600		
# 以下为DMA参数
Y SIZE: 0000000000000001	#1
X SIZE: 0000000000100000	#64	
X STRIDE: 0000000000100000	#64，输出fmap为8×8×32，在内存上连续存放
Y_PAD_0: 0000				#0
Y_PAD_1: 0000				#0
X_PAD_0: 00					#0
X_PAD_1: 00					#0
```



FINISH

该过程并无具体的指令运行，主要通过ARM核来检查四条queue中是否还有缓存标识，如果有，则等待响应的模块进行运算处理，如果没有，则卷积计算过程结束。



## Summary

VTA采用两级ISA的设计，对于某一具体的计算任务，首先，VTA会将该任务的数据缓存方式修改为NCHWnc，随后针对不同FPGA的性能带宽，应用tvm primitives将卷积计算进行分块重排等操作。之后，会生成计算过程完整流程的调度Schedule，VTA会将生成包含API的文件、VTA runtime和预构的bit流文件打包成 .o文件通过RPC发送到板上进行实现。

当VTA在PYNQ上开始运行时，首先会从DRAM中加载计算数据到相应的buffer中。其次，为了保证VTA的GEMM核的高效，VTA一次会加载尽可能多的LOAD、GEMM、STORE指令，尽可能保证GEMM核处于工作状态，同时将LOAD指令插入到GEMM之前实现对数据的读取，这样就不可避免的出现数据相关性的问题。VTA通过对指令加入FEPT FLAGS标识来避免这个问题，通过加入queue消息实现模块之间状态的通信。由于LOAD指令加载时带宽限制会有较长的延时，VTA引入虚拟线程来掩盖DRAM读取的延时，具体已分析过。

