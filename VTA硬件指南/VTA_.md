​	 

 

 

 





# VTA 

 Versatile Tensor Accelerator 通用张量加速器



 

 

 

 

 

 

 

 

























2020/10

------



[TOC]

## Introduction

​	通用张量加速器（Versatile Tensor Accelerator，VTA）是TVM框架的扩展，VTA是一种通用且可编程的加速器，采用类似RISC的抽象化设计，在张量级别进行计算和内存的操作，VTA不仅仅是一个独立的硬件加速器设计，它包含了一个完整的端到端的解决方案，包含了驱动程序、JIT Runtime和基于TVM的用于优化的编译器堆栈，能够实现从深度学习网络模型架构到硬件终端的完整流程。

<img src="C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20201013202726872.png" alt="image-20201013202726872" style="zoom: 33%;" />

​    对某一特定的深度学习网络来说，VTA首先通过TVM将该网络模型转换为Relay IR（特一种特定的编程语言格式），之后Relay会将模型的数据类型进行量化转换为int8。随后Relay IR进行计算图优化，最大化数据重用以及转换inp和wgt数据布局最后转化为TIR，TIR根据已有的算子和schedule方法完成算子的具体实现。之后生成.o文件，加载到目标平台并装载，通过RPC调用启动。

 

##  Design Strategies of VTA

### Programmability

​	VTA的可编程性在于用户可以通过修改`3rdparty/vta-hw/config/vta_config.json`配置文件，用户可以自定义tensor instrinsic的形状，时钟频率，流水线，数据位宽以及片上buffer的尺寸。该文件如下所示，VTA硬件默认INP和WGT位宽为8位，ACC位宽为32位，BATCH=1，默认BLOCK=16，默认UOP BUFFER大小为2^15个字节即32KB，WGT BUFF为256KB，ACC BUFF为128KB。

```shell
{
  "TARGET" : "pynq",			# device target
  "HW_VER" : "0.0.1",			# hardware version 
  "LOG_INP_WIDTH" : 3,			# input data type signed integer width (log2)
  "LOG_WGT_WIDTH" : 3,			# ..
  "LOG_ACC_WIDTH" : 5,			# ..
  "LOG_BATCH" : 0,				# VTA matrix multiply intrinsic input/output dimension 0.(LOG2)
  "LOG_BLOCK" : 4,				# VTA matrix multiply inner dimensions.(LOG2)
  "LOG_UOP_BUFF_SIZE" : 15,		# Micro-op on-chip buffer in Bytes.(LOG2)
  "LOG_INP_BUFF_SIZE" : 15,		# ..
  "LOG_WGT_BUFF_SIZE" : 18,		# ..
  "LOG_ACC_BUFF_SIZE" : 17		# ..
}
```



### Graph Pack

​	在模型经过量化后，VTA会对模型中的数据进行Graph Pack。

> ​	将一个卷积部署在VTA上执行，VTA每个时钟周期将16个不同输入通道的input feature和16个不同输入通道和16个不同输出通道的weight数据进行乘加运算，而通常feature map在内存中存放的格式是NCHW(N:batch, C:channel, H:height, W:width)，即W维度的数据在内存中相邻存放，若要取不同通道的数据，则不是连续的内存访问，降低了内存访问的效率，因此，graph_pack就是对feature和weight在内存中存放的形式进行修改，变为NCHWnc的形式，在N和C的维度对数据进行分组，相邻16通道的数据在内存中连续存放，方便VTA存取。  



### Data loading mode

​	VTA在将数据从DRAM加载到SRAM上时，会采用strided access模式进行加载，如下图所示。这种加载方式直接将DRAM上的数据进行分块和padding，是一种有效的卷积分块策略。考虑到SRAM上的空间有限，必须要将DRAM分块加载，而且当VTA在计算2D卷积的时候不需要再做额外的数据布局操作即可以直接实现input和weight的tile，提高计算效率。

<img src="C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20201013200307631.png" alt="image-20201013200307631" style="zoom:67%;" />

​	load_2d的伪代码如下，sram_base，dram_base，sram_loc，dram_loc均为LOAD指令中的部分字段。

```python
for i in range(0, y + y_pad_0 + y_pad_1):
    for j in range(0, x + x_pad_0 + x_pad_1):
        sram_loc = sram_base + i * (x_size + x_pad_0 + x_pad_1) + j
        dram_loc = dram_base + (i - x_pad_0) * x_stride +(j - y_pad_0)
        if (i < y_pad_0 || i >= y_size + y_pad_0 ||
            j < x_pad_0 || j >= x_size + x_pad_0):
            mem[sram_loc] = 0
        else:
            mem[sram_loc] = DRAM[dram_loc]
```



### Lantency Hiding

​	VTA的各个模块是同步执行的，之间的通信通过Dep Token来完成。为了提高计算资源的利用率，换句话说，要让GEMM尽可能保持在工作状态，VTA引入了Lantency Hiding机制，通过并行化流水线，可以隐藏内存访问带来的延迟。如下图所示，execution savings即为节约下的时间。

Without latency Hiding

<img src="C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20201014110540107.png" alt="image-20201014110540107" style="zoom:50%;" />

After Latency Hiding

<img src="C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20201014110519437.png" alt="image-20201014110519437" style="zoom:50%;" />



### Virtual Threading

​	虚拟线程是一种在VTA硬件设计中增加流水线并行性的机制。换句话说，它通过隐藏内存访问延迟来提高计算资源的利用率。如下图所示，虚拟线程在输出通道上划为为两个线程进行工作，增加了并行度。

<img src="C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20201014110039947.png" alt="image-20201014110039947" style="zoom:50%;" />



## Hardware Architecture

VTA 硬件组织架构如下图所示，主要包括了FETCH、LOAD、COMPUTE和STORE模块，模块之间通过queue和SRAM进行通信，下面将对各个模块进行分析。

<img src="C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20201013170051041.png" alt="image-20201013170051041" style="zoom: 67%;" />

### FETCH Module

​	FETCH模块是从CPU到VTA的接入点，通过以下三个寄存器来实现具体的操作，

​	`Control Reg`：控制FETCH模块的开始和结束，可读可写。

​	`Insn_count Reg`：设置要执行的指令数，只写。

​	`Insns Reg`：设置DRAM中指令流的起始地址，只写。

​	首先，CPU根据VTA runtime在DRAM中准备指令流，随后将起始物理地址写入`Insns`寄存器，并在`Control`中声明起始信号。此过程将启动VTA，通过DMA从DRAM中读取指令流。访问指令流后，FETCH模块将指令解码，并将这些指令推入命令队列中，如下图中LOAD Q、COUMPUTE Q和STORE Q所示。这些命令队列随后将推入LOAD、COMPUTE和STORE模块中。当三个命令队列之一变满时，FETCH模块将暂停工作直到队列未满时。

​	`STORE`指令会被推到STORE Q中由LOAD模块进行处理。

​	`GEMM`和`ALU`指令会被推到COMPUTE Q中由COMPUTE模块进行处理。

​	`LOAD`指令描述微操作内核和寄存器堆文件的指令会被推到COMPUTE Q中由COMPUTE模块进行处理。

​	`LOAD`指令描述加载输入或权重数据的指令才会被推到LOAD Q中。

<img src="C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20201013190319681.png" alt="image-20201013190319681" style="zoom: 50%;" />

### LOAD Module

​	LOAD模块从DRAM到SRAM以strided access模式进行2D DMA传输，执行LOAD Q中的命令将数据迁移到LOAD BUFFER、MICRO-OP SRAM、ACTIVATION SRAM和KERNEL SRAM中。

<img src="C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20201013192744484.png" alt="image-20201013192744484" style="zoom: 50%;" />

### COMPUTE Module

​	VTA的COMPUTE模块与RISC处理器功能类似，主要用于处理张量寄存器的计算部分，主要包括GEMM核和ALU核，COMPUTE模块执行COMPUTE Q中的命令，根据uop执行ALU操作或GEMM操作，如下图所示。

<img src="C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20201013193422875.png" alt="image-20201013193422875" style="zoom: 50%;" />

​	首先，COMPUTE模块从uop cache中读取uop，有两种指令：ALU和GEMM。COMPUTE采用两级循环嵌套的计算模式，以减少调用uop指令的步长，以及避免产生条件跳转指令。GEMM核执行GEMM指令的示意图如下所示，GEMM核每个周期可以执行一个inp-wgt矩阵乘法，uop提供计算数据的存储buffer的地址，其中input存储在inp_mem中，索引为inp_idx；weight存储在wgt_mem中，索引为wgt_idx；计算结果暂存在acc_mem中，索引为acc_idx。



<img src="C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20201013192537916.png" alt="image-20201013192537916" style="zoom:67%;" />

​	 硬件HLS代码中GEMM指令的核心计算部分如下所示，实际完成的操作为：`acc_mem[dst_idx] += GEMM(inp_mem[inp_idx], wgt_mem[wgt_idx])`

```c++
//vta.cc line 287 to 308        
// Inner GEMM loop
for (int b = 0; b < VTA_BATCH; b++) {
	for (int oc = 0; oc < VTA_BLOCK_OUT; oc++) {
    // Initialize the accumulator values
    acc_T accum = a_tensor[b][oc];
    // Dot product sum
    sum_T tmp = 0;
    // Inner matrix multiplication loop (input channel/feature)
    for (int ic = 0; ic < VTA_BLOCK_IN; ic++) {
         wgt_T w_elem = w_tensor[oc][ic];
         inp_T i_elem = i_tensor[b][ic];
         mul_T prod_dsp = i_elem * w_elem;
         tmp += (sum_T) prod_dsp;
       }
       // Update summation
       accum += (acc_T) tmp;
       // Write back result acc_mem
       a_tensor[b][oc] = insn.reset_reg ? (acc_T) 0 : accum;
       // And output vector
       o_tensor[b][oc] = (out_T) accum.range(VTA_OUT_WIDTH - 1, 0);
     }
        
```

​	为了便于理解，GEMM指令的伪代码可以表示为如下所示，end0，end1，uop_bgn，uop_end，x0，x1，y0，y1，z0，z1分别对应到GEMM指令中的字段，具体可以参考下文GEMM指令详解。x，y，z分别表示uop中的字段，即为三种buffer的地址索引。以下三层循环以流水线的方式实现，理论执行时间为`end0*end1*(uop_end-uop_bgn)`个周期。

```python
for i0 in range(0, end0):
	for i1 in range(0, end1):
		for uop_idx in range (uop_bgn, uop_end): 
            # uop指定地址寻址模式
            x, y, z = decode_gemm_indices(uop_buffer[upc]) 
            reg_idx = i0 * x0 + i1 * x1 + x 
            inp_idx = i0 * y0 + i1 * y1 + y 
            wgt_idx = i0 * z0 + i1 * z1 + z
            # 进行矩阵乘法将结果写入acc_mem
			acc_mem[dst_idx] += GEMM(inp_mem[inp_idx], wgt_mem[wgt_idx])
```

​	ALU主要负责执行执行加法，激活，标准化和池化的计算任务。目前支持的算子有MIN，MAX，ADDI，ADD，MULI，MUL，SHLI和SHRI，可以支持张量级别的操作，也可以通过立即数IMM支持标量级别的操作，ALU通过uop进行数据访问。

<img src="C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20201013201100953.png" alt="image-20201013201100953" style="zoom:67%;" />

HLS代码的核心部分如下所示，与GEMM段代码类似，通过uop进行地址索引，再对数据进行相应的计算。

```c++
// Perform ALU op over matrix elements
for (int i = 0; i < VTA_BATCH; i++) {
	for (int b = 0; b < VTA_BLOCK_OUT; b++) {
    // Read in operands
    acc_T src_0 = dst_tensor[i][b];
    acc_T src_1 = insn.use_imm ? (acc_T) insn.imm : src_tensor[i][b];
    aluop_shr_arg_T shft_by = src_1.range(VTA_SHR_ARG_BIT_WIDTH - 1, 0);
    aluop_mul_arg_T mul_by = src_1.range(VTA_MUL_ARG_BIT_WIDTH - 1, 0);
    if (insn.alu_opcode == VTA_ALU_OPCODE_MIN || insn.alu_opcode == VTA_ALU_OPCODE_MAX) {
    // Compute Min/Max
    acc_T mix_val = src_0 < src_1 ?
          (insn.alu_opcode == VTA_ALU_OPCODE_MIN ? src_0 : src_1) :
          (insn.alu_opcode == VTA_ALU_OPCODE_MIN ? src_1 : src_0);
    dst_tensor[i][b] = mix_val;
    o_tensor[i][b] = (out_T) mix_val.range(VTA_OUT_WIDTH - 1, 0);
   } else if (insn.alu_opcode == VTA_ALU_OPCODE_ADD) {
     // Compute Sum
     acc_T add_val =
         src_0.range(VTA_ACC_WIDTH - 1, 0) + src_1.range(VTA_ACC_WIDTH - 1, 0);
     dst_tensor[i][b] = add_val;
     o_tensor[i][b] = (out_T) add_val.range(VTA_OUT_WIDTH - 1, 0);
   } else if (insn.alu_opcode == VTA_ALU_OPCODE_SHR) {
     // Compute Shift Right
     acc_T shr_val = src_0 >> shft_by;
     dst_tensor[i][b] = shr_val;
     o_tensor[i][b] = (out_T) shr_val.range(VTA_OUT_WIDTH - 1, 0);
    }
  }
}
```

为了便于理解，ALU指令的伪代码如下给出，与GEMM类似也是分为三层循环，硬件上通过流水线的方式实现，具体不再赘述。

```python
for i0 in range(0, end0):
    for i1 in range(0, end1):
        for uop_idx in range (uop_bgn, uop_end):
            x,y = decode_alu_indices(uop_buffer[upc])
            dst_idx = i0 * x0 + i1 * x1 + x
            stc_idx = i0 * y0 + i1 * y1 + y
            if USE_IMM:
                reg_file[dst_idx] = OP(reg_file[dst_idx], IMM)
            else:
                reg_file[dst_idx] = OP(reg_file[dst_idx], reg_file[src_idx])
```



### STORE Module

 	与LOAD模块功能相反，根据STORE Q中的命令将store buffer中的数据写回DRAM中，如下图所示。

<img src="C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20201013193906319.png" alt="image-20201013193906319" style="zoom: 50%;" />

### Dataflow Execution

​	VTA中所用到的片上缓存如下图所示，包括LOAD BUFFER、MICRO-OP SRAM、ACTIVATION SRAM、KERNEL SRAM和STORE BUFFER。这些BUFFER是由模块共享的，由于三个模块是同时运行的，所以不可避免的会产生数据冲突，VTA为了解决这个问题，引入了Dependence Token来解决。

<img src="C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20201014094832892.png" alt="image-20201014094832892" style="zoom:50%;" />

​	硬件各个模块之间通过Dependence Token Queues进行通信，以指示各个模块将要执行处理的命令。这种dep token的本质上是FIFO，下图简化了VTA的硬件模型，显示了队列通信之间的具体关系，每个模块通过写后读（RAW）和读后写（WAR）的dep queue连接到各个模块。

> ​	由于上面指出，只有LOAD和COMPUTE、COMPUTE和STORE之间存在共享BUFFER，因此只需要在LOAD和COMPUTE、COMPUTE和STORE之间加入dependency queue，而LOAD和STORE之间不需要。假设一条LOAD指令和一条COMPUTE指令之间存在数据相关性(RAW)，即数据被LOAD写到BUFFER中
> 后，才能被COMPUTE读取去计算。而由于LOAD和COMPUTE指令会同时到达执行部件，为了解决数据冲突，这两条指令必须先执行完LOAD再执行COMPUTE。因此在指令中做这样的设置，LOAD指令在数据读取至BUFFER后，向LD->CMP Q中写入1，COMPUTE在指令执行前一直查询LD->CMP Q，直至其非空且能读出1（意味着LOAD指令执行完毕），才开始执行COMPUTE指令。这样就解决了数据相关的问题。  
>
> ​	指令是否需要读取或写入dependency queue由指令字中的某些位决定，本质上说，数据冲突的解决是在指令生成时解决的。更普遍地讲，指令被发送到各个CMD Q之后，对于同一执行部件，指令会顺序执行。数据相关信息会被按顺序写入dependency queue，也会被顺序读取。  

<img src="C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20201014101108260.png" alt="image-20201014101108260" style="zoom:50%;" />

​	队列通信的相应伪代码如下所示，该过程基于对其他指令的进程而确定。首先，每个指令内的DEPT FLAGS都在硬件中解码，如果执行的指令具有传入的RAW Dependence，则根据producer模块接受到的RAW Dep Token来确定模块是否工作。类似的，如果执行的指令具有传入的WAR Dependence，则根据从comsumer接受到的WAR Dep token来确定是否继续工作。最后，当所有指令完成时，会检查队列中是否还有Dep Token，清空后会通知所有模块停止工作。

```python
insn_T insn = cmd.pop()
bool wait_on_producer = insn.pop_prev
bool wait_on_consumer = insn.pop_next
bool notify_producer = insn.push_prev
bool notify_consumer = insn.push_next
if wait_on_producer:
	while producer_raw_queue.empty() ship
	producer_raw_queue.pop()
if wait_on_consumer：
	while consumer_war_queue.empty() skip
	consumer_war_queue.pop()
	
# Do task

if notify_producer:
	producer_war_queue.push(1)
if notify_consumer:
	consumer_raw_queue.push(1)
```



## VTA Instructions

### Repository Overview

https://github.com/apache/incubator-tvm-vta

```python
driver
	pynq_driver.cc      	# PYNQ board drivers source file 
	pynq_drive.h      		# PYNQ board drivers header file

scripts
    hls.tcl              	# HLS tcl scripts: sim, syn, ip generation
    hsi.tcl           		# HIS tcl scripts: generates ARM device drivers
    vivado.tcl        		# Vivado tcl scripts: run logic syn, place, route, gen vta.bit

src
    hw_spec.h         		# hardware spec
    test_lib.cc         	# test lib src file
    test_lib.h         		# test lib header file
    vta.cc            		# VTA HLS-based hardware src file
    vta.h             		# VTA HLS-based hardware header file
    vta_test.cc       		# test harness for sim and hardware

Makefile            		# Makefile for sim, hardware compilation and device test.
```

​	VTA的设计采用两级指令集架构（Instruction Set Architecture, ISA）, 包括一个任务级ISA，该ISA明确协调并发的计算和内存任务，和一个微代码ISA，该ISA存放在微操作cache中，用于具体描述如何对数据进行计算，不提供控制流。

任务级ISA主要包括LOAD、GEMM、ALU、和STORE指令，如下图所示。LOAD和STORE指令描述了如何将DRAM中的数据加载并存储到片上SRAM中，GEMM和ALU指令根据uop指令调用相应的算子对相应的数据进行计算。

指令与模块的关系如下：

​	FETCH模块从DRAM中加载任务级ISA，根据指令的类型将它们分配到LOAD、COMPUTE和STORE模块相应的command queues

​	LOAD模块从DRAM中加载input、weight、bias数据到片上内存中。

​	COMPUTE模块读取队列和指令中的信息，通过GEMM内核执行二维卷积、全连接运算，通过ALU执行加法，激活函数，归一化和池化操作。

​	STORE模块从COMPUTE中读取计算结果并写入DRAM。



### LOAD,STORE 

![](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20201013161255892.png)

```python
LOAD，STORE
Usage：perform 2D strided DMA reads/writes between DRAM and SRAM.

OPCODE                  	# instruction opcode
DEPT FLAGS               	# dependence token，4位
BUFFER ID                   # Source/destination SRAM for store/load instruction，
SRAM BASE                   # SRAM base address
DRAM BASE                   # DRAM base address
Y SIZE                      # 2D access pattern: y-size
X SIZE                      # 2D access pattern: x-size
X STRIDE                    # 2D access pattern: x-stride
Y PAD: TOP                  # 2D access pattern: start padding along y dimension
Y PAD: BOTTOM               # 2D access pattern: end padding along y dimension
X PAD: LEFT                 # 2D access pattern: start padding along x dimension
X PAD: RIGHT                # 2D access pattern: end padding along x dimension
```



### GEMM ![image-20201013162707656](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20201013162707656.png)

```python
GEMM
Usage：used to matrix multiplication, 2D convolutions.

OPCODE                     #
DEPT FLAGS                 # dep token
RESET                      # reset，1位
MICRO-OP BEGIN             # Micro-op begin address
MICRO-OP END               # Micro-op end address
LOOP EXTENT 0              # Iterations in the outer uop execution loop
LOOP EXTENT 1              # Iterations in the inner uop execution loop
ACCUM IDX FACTOR 0         # Outer loop accumulator memory index factor
ACCUM IDX FACTOR 1         # Inner loop accumulator memory index factor
INPUT IDX FACTOR 0         # Outer loop input memory index factor
INPUT IDX FACTOR 1         # Inner loop input memory index factor
WEIGHT IDX FACTOR 0        # Outer loop weight memory index factor
WEIGHT IDX FACTOR 1        # Inner loop weight memory index factor
```

​	整个VTA的设计核心是围绕GEMM核展开的，相应的计算数据存放在片上SRAM BUFFER中，用到了以下四种：

   	 inp_mem：存储输入数据，R-only。每个周期读取一个（VTA_BATCH，VTA_BLOCK_IN）大小的输入矩阵。

 	   wgt_mem：存储权重数据，R-only。每个周期读取一个（VTA_BLOCK_OUT，VTA_BLOCK_IN）大小的权重矩阵。

 	   acc_mem：累加器数据的读写，R/W。每个周期读取一个（VTA_BATCH，VTA_BLOCK_OUT）大小的矩阵。

   	 uop_mem：uop指令存储，R-only。每个周期uop从uop_mem中读取，uop由以下三部分构成：	 

```
   dst_idx：index of acc_mem
   src_idx：index of inp_mem
   wgt_idx：index of wgt_mem
```

​    由uop指定的矩阵乘法运算即为：

```python
   acc_mem[dst_idx] += GEMM(inp_mem[src_idx], wgt_mem[wgt_idx])
```



### ALU 

![image-20201013162956923](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20201013162956923.png)

```python
ALU
Usage：perform a wide range of activation, normalization, and pooling tasks.

OPCODE                     # ..
DEPT FLAGS                 # ..
RESET                      # ..
MICRO-OP BEGIN             # ..
MICRO-OP END               # ..
LOOP EXTENT 0              # ..
LOOP EXTENT 1              # ..
DST IDX FACTOR 0           # Outer loop accumulator memory destination index factor
DST IDX FACTOR 1           # Inner loop accumulator memory destination index factor
SRC IDX FACTOR 0           # Outer loop accumulator memory source index factor
SRC IDX FACTOR 1           # Inner loop accumulator memory source index factor
ALU OPCODE                 # opcode
USE_IMM                    # Whether use IMM or not
IMMEDIATE                  # Immediate value
```

 

## CONV2D Analysis

> VTA硬件参数如下：
>
> VTA_BATCH = 1
> VTA_BLOCK_OUT= 32
> VTA_BLOCK_IN = 32
>
> 卷积形状为：
>
> ```
> Conv2DWorkload(batch=1, height=8, width=8, in_filter=32, out_filter=32,
> hkernel=3, wkernel=3, hpad=1, wpad=1, hstride=1, wstride=1)  
> ```

​	VTA的两级指令架构，第一级指令架构使用CISC-ness来描述高层次操作，包括上文中提到的LOAD、STORE、GEMM和ALU。

​	第二级指令架构采用RISC-ness来描述低层次内存访问访问模式，称为MRCIO-OP，简称为uop，每条GEMM/ALU指令对应一组uop。

​	对于GEMM指令的uop来说，其结构为（accumulator index（acc），input index（inp），weight index（wgt））。共32位，不足部分补零。

上述卷积打印出来的指令如下所示：

```shell
# 为了便于区分，以下定义为uop_a
# 对应INSTRUCTION 1, reset操作
There are 1 uops				
[0000] acc=0, inp=0, wgt=0

# 对应到INSTRUCTION 4,提供acc,inp和wgt的索引
# uop_b
There are 24 uops
[0000] acc=0, inp=0, wgt=0
[0001] acc=8, inp=10, wgt=0
[0002] acc=16, inp=20, wgt=0
[0003] acc=24, inp=30, wgt=0
[0004] acc=32, inp=40, wgt=0
[0005] acc=40, inp=50, wgt=0
[0006] acc=48, inp=60, wgt=0
[0007] acc=56, inp=70, wgt=0
[0008] acc=0, inp=10, wgt=3
[0009] acc=8, inp=20, wgt=3
[0010] acc=16, inp=30, wgt=3
[0011] acc=24, inp=40, wgt=3
[0012] acc=32, inp=50, wgt=3
[0013] acc=40, inp=60, wgt=3
[0014] acc=48, inp=70, wgt=3
[0015] acc=56, inp=80, wgt=3
[0016] acc=0, inp=20, wgt=6
[0017] acc=8, inp=30, wgt=6
[0018] acc=16, inp=40, wgt=6
[0019] acc=24, inp=50, wgt=6
[0020] acc=32, inp=60, wgt=6
[0021] acc=40, inp=70, wgt=6
[0022] acc=48, inp=80, wgt=6
[0023] acc=56, inp=90, wgt=6

#uop_c
There are 1 uops
[0000] acc=0, inp=0, wgt=0

#uop_d
There are 1 uops
[0000] acc=0, inp=64, wgt=0

#uop_e
There are 1 uops
[0000] acc=0, inp=0, wgt=0

#uop_f
There are 1 uops
[0000] acc=0, inp=0, wgt=0


# param explanation
# DEPT FLAGS, 每条指令中自带的标识。
# pop prev: pop dependence from previous stage, 即等待LOAD模块加载完成的标识，位于LOAD→COM Q中。
# pop next: pop dependence from next stage. 即等待STORE模块存储完成的标识，位于STORE→COM Q中。
# push prev: push dependence from previous stage. 即通知LOAD模块计算完成的标识，位于COM→LOAD Q中。
# push next: push dependence from next stage. 即通知SROTE模块完成计算的标识，位于COM→STORE Q中。
#
# y size: Number of rows. 
# x size: Number of columns. 
# stride: Stride along the x axis
# 
#
# 用于检测FIFO队列中中FLAG的数目，在调度过程以及结束时避免出现数据冲突。
# l2g_queue: load to gemm queue.
# g2l_queue: gemm to load queue.
# s2g_queue: store to gemm queue.
# g2s_queue: gemm to store queue.

There are 19 instructions

# COMPUTE模块加载uop,对gemm指令进行reset操作。
INSTRUCTION 0: LOAD UOP
dep - pop prev: 0, pop next: 0, push prev: 0, push next: 0
# DRAM和SRAM的地址
DRAM: 0x00001c00, SRAM:0x0000
y: size=1, pad=[0, 0]
x: size=1, stride=1, pad=[0, 0]
l2g_queue = 0, g2l_queue = 0
s2g_queue = 0, g2s_queue = 0

# uop加载后为GEMM指令，此时进行reset_out=1,进行reset操作
INSTRUCTION 1: GEMM
dep - pop prev: 0, pop next: 0, push prev: 1, push next: 0
reset_out: 1
# range标识uop取址范围，对应到uop_a
range (0, 1)
outer loop - iter: 8, wgt: 0, inp: 0, acc: 1
inner loop - iter: 8, wgt: 0, inp: 0, acc: 8
l2g_queue = 0, g2l_queue = 1
s2g_queue = 0, g2s_queue = 0

# LOAD指令，加载INP数据，同时硬件将dep token译码并写入FIFO
INSTRUCTION 2: LOAD INP
dep - pop prev: 0, pop next: 1, push prev: 0, push next: 0
DRAM: 0x00000100, SRAM:0x0000
# 经过grap pack之后，data_layout变成NCHWnc,相当于对32个输入通道进行打包，卷积计算在一个周期内进行。
# 此时inp_mem中数据布局为（N, IC, H, W, n, ic），即为（1，1，8，8，1，32）
# wgt_mem中数据布局为（OC, IC, H, W, oc, ic），即为（1，1，3，3，32，32）
# acc_mem中数据布局为（N, OC, H, W, n, oc），即为（1，1，8，8，1，32）
# 由于h=w=8，h_pad=w_pad=1,即在load_2d中每次加载（8，8）大小的数据，同时进行pad=[1,1]的操作。
y: size=8, pad=[1, 1]
x: size=8, stride=8, pad=[1, 1]
l2g_queue = 0, g2l_queue = 0
s2g_queue = 0, g2s_queue = 0

# 加载WGT数据，写入dep flags
INSTRUCTION 3: LOAD WGT
dep - pop prev: 0, pop next: 0, push prev: 0, push next: 1
DRAM: 0x00000020, SRAM:0x0000
y: size=1, pad=[0, 0]
x: size=9, stride=9, pad=[0, 0]
l2g_queue = 1, g2l_queue = 0
s2g_queue = 0, g2s_queue = 0

# 对应到uop_b
INSTRUCTION 4: LOAD UOP
dep - pop prev: 1, pop next: 0, push prev: 0, push next: 0
DRAM: 0x00001c01, SRAM:0x0001
y: size=1, pad=[0, 0]
x: size=24, stride=24, pad=[0, 0]
l2g_queue = 0, g2l_queue = 0
s2g_queue = 0, g2s_queue = 0

# uop共循环了24次，GEMM指令一共进行8*3*24次循环
INSTRUCTION 5: GEMM
dep - pop prev: 0, pop next: 0, push prev: 0, push next: 0
reset_out: 0
range (1, 25)
outer loop - iter: 8, wgt: 0, inp: 1, acc: 1
inner loop - iter: 3, wgt: 1, inp: 1, acc: 0
l2g_queue = 0, g2l_queue = 0
s2g_queue = 0, g2s_queue = 0

# 对应到uop_c,推入ALU进行移位操作
INSTRUCTION 6: LOAD UOP
dep - pop prev: 0, pop next: 0, push prev: 0, push next: 0
DRAM: 0x00001c19, SRAM:0x0019
y: size=1, pad=[0, 0]
x: size=1, stride=1, pad=[0, 0]
l2g_queue = 0, g2l_queue = 0
s2g_queue = 0, g2s_queue = 0

# ALU SHR
INSTRUCTION 7: ALU - shr
dep - pop prev: 0, pop next: 0, push prev: 0, push next: 0
reset_out: 0
range (25, 26)
outer loop - iter: 1, dst: 0, src: 0
inner loop - iter: 64, dst: 1, src: 1
l2g_queue = 0, g2l_queue = 0
s2g_queue = 0, g2s_queue = 0

INSTRUCTION 8: LOAD ACC
dep - pop prev: 0, pop next: 0, push prev: 0, push next: 0
DRAM: 0x00000140, SRAM:0x0040
y: size=1, pad=[0, 0]
x: size=1, stride=1, pad=[0, 0]
l2g_queue = 0, g2l_queue = 0
s2g_queue = 0, g2s_queue = 0

# 对应到uop_d
INSTRUCTION 9: LOAD UOP
dep - pop prev: 0, pop next: 0, push prev: 0, push next: 0
DRAM: 0x00001c1a, SRAM:0x001a
y: size=1, pad=[0, 0]
x: size=1, stride=1, pad=[0, 0]
l2g_queue = 0, g2l_queue = 0
s2g_queue = 0, g2s_queue = 0

# ALU ADD
INSTRUCTION 10: ALU - add
dep - pop prev: 0, pop next: 0, push prev: 0, push next: 0
reset_out: 0
range (26, 27)
outer loop - iter: 1, dst: 0, src: 0
inner loop - iter: 64, dst: 1, src: 0
l2g_queue = 0, g2l_queue = 0
s2g_queue = 0, g2s_queue = 0

# 对应到uop_e
INSTRUCTION 11: LOAD UOP
dep - pop prev: 0, pop next: 0, push prev: 0, push next: 0
DRAM: 0x00001c1b, SRAM:0x001b
y: size=1, pad=[0, 0]
x: size=1, stride=1, pad=[0, 0]
l2g_queue = 0, g2l_queue = 0
s2g_queue = 0, g2s_queue = 0

# MIN
INSTRUCTION 12: ALU - min imm
dep - pop prev: 0, pop next: 0, push prev: 0, push next: 0
reset_out: 0
range (27, 28)
outer loop - iter: 1, dst: 0, src: 0
inner loop - iter: 64, dst: 1, src: 1
l2g_queue = 0, g2l_queue = 0
s2g_queue = 0, g2s_queue = 0

# 对应到uop_f
INSTRUCTION 13: LOAD UOP
dep - pop prev: 0, pop next: 0, push prev: 0, push next: 0
DRAM: 0x00001c1c, SRAM:0x001c
y: size=1, pad=[0, 0]
x: size=1, stride=1, pad=[0, 0]
l2g_queue = 0, g2l_queue = 0
s2g_queue = 0, g2s_queue = 0

# MAX
INSTRUCTION 14: ALU - max imm
dep - pop prev: 0, pop next: 0, push prev: 0, push next: 1
reset_out: 0
range (28, 29)
outer loop - iter: 1, dst: 0, src: 0
inner loop - iter: 64, dst: 1, src: 1
l2g_queue = 0, g2l_queue = 0
s2g_queue = 0, g2s_queue = 1

# 将结果写回DRAM
INSTRUCTION 15: STORE:
dep - pop prev: 1, pop next: 0, push prev: 1, push next: 0
DRAM: 0x00000600, SRAM:0x0000
y: size=1, pad=[0, 0]
x: size=64, stride=64, pad=[0, 0]
l2g_queue = 0, g2l_queue = 0
s2g_queue = 1, g2s_queue = 0

# 检查dep token，清空buffer
INSTRUCTION 16: NOP-MEMORY-STAGE
dep - pop prev: 0, pop next: 0, push prev: 0, push next: 1
l2g_queue = 1, g2l_queue = 0
s2g_queue = 1, g2s_queue = 0

INSTRUCTION 17: NOP-COMPUTE-STAGE
dep - pop prev: 1, pop next: 1, push prev: 0, push next: 0
l2g_queue = 0, g2l_queue = 0
s2g_queue = 0, g2s_queue = 0

# finish，FIFO清零
INSTRUCTION 18: FINISH
l2g_queue = 0, g2l_queue = 0
s2g_queue = 0, g2s_queue = 0
```



## Q&A

1.VTA的本质是什么？

答：VTA本质上是一种基于TVM平台，将深度学习模型部署到FPGA上的硬件加速器。



2.VTA中有哪些指令？

答：VTA采用两级ISA设计，第一级指令包括LOAD、GEMM、ALU和STORE，长度为128位，主要对数据提供控制流，协调并发的计算任务。第二级指令uop为32位，主要提供数据的地址，具体描述如何对数据进行计算。



3.uop指令是怎么组成的，存放在哪里？

答：如下图所示，uop指令存放在uop buffer中，由acc_idx、inp_idx和wgt_idx这三部分构成。

![image-20201019141450077](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20201019141450077.png)

4.两级指令之间的关系？

答：在计算过程中，每条uop指令都会对应到一条GEMM或ALU指令。以CONV2D中instruction：4和instruction：5为例GEMM指令为例，uop指令首先会从uop buffer中取出需要操作的数据并发送到相应的buffer中，随后，GEMM指令根据接收到的数据进行计算，并将计算的结果发送到acc buffer中。

```shell
# 对应到INSTRUCTION 4,提供acc,inp和wgt的索引
# acc、inp、wgt分别表示上图中的acc_idx、inp_idx和wgt_idx
# outer loop和iner loop共8*3次循环，所以uop要取址24次
# uop_b
There are 24 uops
[0000] acc=0, inp=0, wgt=0
[0001] acc=8, inp=10, wgt=0
[0002] acc=16, inp=20, wgt=0
[0003] acc=24, inp=30, wgt=0
[0004] acc=32, inp=40, wgt=0
[0005] acc=40, inp=50, wgt=0
[0006] acc=48, inp=60, wgt=0
[0007] acc=56, inp=70, wgt=0
[0008] acc=0, inp=10, wgt=3
[0009] acc=8, inp=20, wgt=3
[0010] acc=16, inp=30, wgt=3
[0011] acc=24, inp=40, wgt=3
[0012] acc=32, inp=50, wgt=3
[0013] acc=40, inp=60, wgt=3
[0014] acc=48, inp=70, wgt=3
[0015] acc=56, inp=80, wgt=3
[0016] acc=0, inp=20, wgt=6
[0017] acc=8, inp=30, wgt=6
[0018] acc=16, inp=40, wgt=6
[0019] acc=24, inp=50, wgt=6
[0020] acc=32, inp=60, wgt=6
[0021] acc=40, inp=70, wgt=6
[0022] acc=48, inp=80, wgt=6
[0023] acc=56, inp=90, wgt=6

# 对应到uop_b
INSTRUCTION 4: LOAD UOP
dep - pop prev: 1, pop next: 0, push prev: 0, push next: 0
DRAM: 0x00001c01, SRAM:0x0001
y: size=1, pad=[0, 0]
x: size=24, stride=24, pad=[0, 0]
l2g_queue = 0, g2l_queue = 0
s2g_queue = 0, g2s_queue = 0

# uop共循环了24次，GEMM指令一共进行8*3*24次循环
INSTRUCTION 5: GEMM
dep - pop prev: 0, pop next: 0, push prev: 0, push next: 0
reset_out: 0
range (1, 25)
outer loop - iter: 8, wgt: 0, inp: 1, acc: 1
inner loop - iter: 3, wgt: 1, inp: 1, acc: 0
l2g_queue = 0, g2l_queue = 0
s2g_queue = 0, g2s_queue = 0  
```



5.什么是pop prev / pop next ... /push next？

答：这四个参数对应到LOAD、GEMM、ALU和STORE指令字段中的DEPT FLAGS，表示LOAD、COMPUTE和STORE模块之间的通信关系，由于三者是同步执行的关系，相互之间会存在数据相关性的问题，这四个FLAGS就是用来产生相应的标识，存放在相应的队列中，只有当GEMM指令完成运算后，才会送到STORE模块中，同时LOAD模块加载好下一次计算的数据才可以加载到COMPUTE模块中。



6.什么是l2g_queue / ... / g2s_queue？

答：如之前所提到的，LOAD模块与COMPUTE模块、COMPUTE模块与STORE模块之间会有数据相关性的问题，加入了DEPT FLAGS用于标识当前模块的状态。上述4个queue便是用于检测DEPT FLAGS中的状态，当queue中所有状态清空时，该计算过程才finish。



7.数据在VTA中是怎么存放的，具体有哪些buffer？

答：主要有inp_mem(inp buffer，int8)、wgt_mem(wgt buffer，int8)、acc_mem(acc buffer，int32)和uop_mem(uop buffer，32位)。具体作用在VTA Instruction GEMM中已提及。



## References

[1] Moreau T , Chen T , Vega L , et al. A Hardware-Software Blueprint for Flexible Deep Learning Specialization[J]. 2018.

[2] 常蕃. Conv2D+elewise_op schedule. 2020.

[3] https://github.com/apache/incubator-tvm-vta

[4] https://tvm.apache.org/docs/vta/index.html