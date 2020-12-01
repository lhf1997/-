/*
load 模块测试代码

2020/11

*/

`timescale 1ns/1ps
module tb_load();
	reg ap_clk, ap_rst_n;
  
	parameter clockperiod = 10;
  
	initial begin
		ap_clk = 0;
		forever #(clockperiod/2) ap_clk = ~ap_clk;
	end
	
	//s_axi_CONTROL_BUS
	reg [4:0] s_axi_CONTROL_BUS_AWADDR;
	reg s_axi_CONTROL_BUS_AWVALID;
	wire s_axi_CONTROL_BUS_AWREADY;
	reg [31:0] s_axi_CONTROL_BUS_WDATA;
	reg [3:0] s_axi_CONTROL_BUS_WSTRB;
	reg s_axi_CONTROL_BUS_WVALID;
	wire s_axi_CONTROL_BUS_WREADY;
	wire [1:0] s_axi_CONTROL_BUS_BRESP;
	wire s_axi_CONTROL_BUS_BVALID;
	reg s_axi_CONTROL_BUS_BREADY;
	
	reg [4:0] s_axi_CONTROL_BUS_ARADDR;
	reg s_axi_CONTROL_BUS_ARVALID;
	wire s_axi_CONTROL_BUS_ARREADY;
	wire [31:0] s_axi_CONTROL_BUS_RDATA;
	wire [1:0] s_axi_CONTROL_BUS_RRESP;
	wire s_axi_CONTROL_BUS_RVALID;
	reg s_axi_CONTROL_BUS_RREADY;
	
	// m_axi_data_port
	wire [31:0] m_axi_data_port_AWADDR;
	wire [7:0] m_axi_data_port_AWLEN;
	wire [2:0] m_axi_data_port_AWSIZE;
	wire m_axi_data_port_AWVALID;
	reg m_axi_data_port_AWREADY;
	wire [64:0] m_axi_data_port_WDATA;
	wire [7:0] m_axi_data_port_WSTRB;
	wire m_axi_data_port_WLAST;
	wire m_axi_data_port_WVALID;
	reg m_axi_data_port_WREADY;
	reg [1:0] m_axi_data_port_BRESP;
	reg m_axi_data_port_BVALID;
	wire m_axi_data_port_BREADY;
	wire [31:0] m_axi_data_port_ARADDR;
	wire m_axi_data_port_ARVALID;
	reg m_axi_data_port_ARREADY;
	reg [63:0] m_axi_data_port_RDATA;
	reg [1:0] m_axi_data_port_RRESP;
	reg m_axi_data_port_RLAST;
	reg m_axi_data_port_RVALID;
	wire m_axi_data_port_RREADY;
	
	//load_queue
	reg load_queue_V_V_TVALID;
	wire load_queue_V_V_TREADY;
	reg [127:0] load_queue_V_V_TDATA;
	
	//l2g_dep_queue  
	wire [7:0] l2g_dep_queue_V_TDATA;
	wire l2g_dep_queue_V_TVALID;
	reg l2g_dep_queue_V_TREADY;
	
	//g2l_dep_queue
	reg g2l_dep_queue_V_TVALID;
	wire g2l_dep_queue_V_TREADY;
	reg [7:0] g2l_dep_queue_V_TDATA;

	//inp_mem
	wire [15:0] inp_mem_V_WEN_A;
	wire [31:0] inp_mem_V_Addr_A;
	wire [127:0] inp_mem_V_Din_A;
	wire [127:0] inp_mem_V_Dout_A;
	

	
    wire [127:0] insn;
	
	// s_axi_CONTROL_BUS
	initial begin 
		ap_rst_n = 1'b0;
		s_axi_CONTROL_BUS_AWADDR = 0;
		s_axi_CONTROL_BUS_AWVALID = 0;
		
		s_axi_CONTROL_BUS_WDATA = 0;
		s_axi_CONTROL_BUS_WSTRB = 0;
		s_axi_CONTROL_BUS_WVALID = 0;
		
		s_axi_CONTROL_BUS_BREADY = 1;
		
		repeat(3)
		@(negedge ap_clk); 
		ap_rst_n = 1'b1;
		
		@(posedge ap_clk);
		s_axi_CONTROL_BUS_AWADDR <= 4'b0;
		s_axi_CONTROL_BUS_AWVALID <= 1'b1;
		@(posedge ap_clk);
		s_axi_CONTROL_BUS_AWVALID <= 1'b0;
		
		s_axi_CONTROL_BUS_WDATA <= 32'h81;
		s_axi_CONTROL_BUS_WSTRB <= 4'b1111;
		s_axi_CONTROL_BUS_WVALID <= 1'b1;
		@(posedge ap_clk);
		s_axi_CONTROL_BUS_WVALID <= 1'b0;
		
	end
	
	// load_queue 
	initial begin
		load_queue_V_V_TVALID = 0;
		load_queue_V_V_TDATA = 0;
		
		repeat(5) 
		@(posedge ap_clk);
		load_queue_V_V_TDATA <= insn;
		load_queue_V_V_TVALID <= 1'b1;
		
		@(posedge ap_clk);
		load_queue_V_V_TVALID <= 1'b0;
	end
  
	// instructions parameter
	parameter OPCODE_BIT_WIDTH = 3;
	parameter MEMOP_ID_BIT_WIDTH = 2;
	parameter MEMOP_SRAM_ADDR_WIDTH = 16;
	parameter MEMOP_DRAM_ADDR_WIDTH = 32;
	parameter MEMOP_SIZE_BIT_WIDTH = 16;
	parameter MEMOP_STRIDE_BIT_WIDTH = 16;
	parameter MEMOP_PAD_BIT_WIDTH = 4;
	
	parameter OPCODE_LOAD = 0;
	parameter OPCODE_STORE = 1;
	parameter OPCODE_GEMM = 2;
	parameter OPCODE_FINISH = 3;
	parameter OPCODE_ALU = 4;
	
	parameter MEM_ID_UOP = 0;
	parameter MEM_ID_WGT = 1;
	parameter MEM_ID_INP = 2;
	parameter MEM_ID_ACC = 3;
	parameter MEM_ID_OUT = 4;
	
	reg [OPCODE_BIT_WIDTH-1:0] opcode;
	reg [3:0] dep;
	reg [MEMOP_ID_BIT_WIDTH-1:0] buffer_id;
	reg [MEMOP_SRAM_ADDR_WIDTH-1:0] sram_base;
	reg [MEMOP_DRAM_ADDR_WIDTH-1:0] dram_base;
	reg [64-OPCODE_BIT_WIDTH-4-MEMOP_ID_BIT_WIDTH-MEMOP_SRAM_ADDR_WIDTH-MEMOP_DRAM_ADDR_WIDTH-1:0] unused_1;
	reg [MEMOP_SIZE_BIT_WIDTH-1:0] y_size, x_size;
	reg [MEMOP_STRIDE_BIT_WIDTH-1:0] x_stride;
	reg [MEMOP_PAD_BIT_WIDTH-1:0] y_pad_0, y_pad_1, x_pad_0, x_pad_1;
	// reg [64-2*MEMOP_SIZE_BIT_WIDTH-MEMOP_STRIDE_BIT_WIDTH-4*MEMOP_PAD_BIT_WIDTH-1:0] unused_2;
	
	assign insn = {/*unused_2, */x_pad_1, x_pad_0, y_pad_1, y_pad_0, x_stride, x_size, y_size, unused_1, dram_base, sram_base, buffer_id, dep, opcode};
	
	initial begin 
		opcode = OPCODE_LOAD;
		dep = 4'b1010;
		buffer_id = MEM_ID_INP;
		sram_base = 1;
		dram_base = 1;
		y_size = 1;
		x_size = 16;
		x_stride = 0;
		{x_pad_1, x_pad_0, y_pad_1, y_pad_0} = 4'h0000;
		unused_1 = 0;
		// unused_2 = 0;
		
		repeat(10)
		@(posedge ap_clk);
		opcode = OPCODE_LOAD;
		dep = 4'b1010;
		buffer_id = MEM_ID_WGT;
		sram_base = 1;
		dram_base = 2;
		y_size = 1;
		x_size = 1;
		x_stride = 0;
		{x_pad_1, x_pad_0, y_pad_1, y_pad_0} = 4'h0000;
		unused_1 = 0;
		// unused_2 = 0;
	end

	/*  
	*/
	integer i;
	initial begin
		m_axi_data_port_ARREADY = 0;
		m_axi_data_port_RDATA = 0;
		m_axi_data_port_RVALID = 0;
		m_axi_data_port_RLAST = 0;
		m_axi_data_port_RRESP = 0;
		
		repeat(7)
		@(posedge ap_clk);
		m_axi_data_port_ARREADY <= 1;
		
        @(negedge m_axi_data_port_ARVALID);
		//@(posedge ap_clk);
		  m_axi_data_port_RVALID <= 1'b1;
		  m_axi_data_port_RDATA <= 'd1;
		  for(i=1;i<32;i=i+1) begin 
			  wait(m_axi_data_port_RVALID && m_axi_data_port_RREADY);
			  @(posedge ap_clk);
			  m_axi_data_port_RDATA <= i + 1; 
			  if(i==15) begin
			  m_axi_data_port_RLAST <= 1;
			  @(posedge ap_clk);
			  m_axi_data_port_RLAST <=0; end
			  else if(i==31) begin
			  m_axi_data_port_RLAST <= 1;
			  @(posedge ap_clk);
			  m_axi_data_port_RLAST <=0;
			  m_axi_data_port_RVALID <= 0;
			  end		  
			end
		
		wait(m_axi_data_port_RVALID && m_axi_data_port_RREADY);
		@(posedge ap_clk);
		m_axi_data_port_ARREADY <= 0;
		
		repeat(13)
		@(posedge ap_clk);
		m_axi_data_port_ARREADY <= 1;
	end
	
	initial begin
	
		repeat(41) 
		@(posedge ap_clk);
		load_queue_V_V_TDATA <= insn;
		load_queue_V_V_TVALID <= 1'b1;
		
		@(posedge ap_clk);
		load_queue_V_V_TVALID <= 1'b0;
		m_axi_data_port_ARREADY <= 1;
		
		@(negedge m_axi_data_port_ARVALID);
		//@(posedge ap_clk);
		  m_axi_data_port_RVALID <= 1'b1;
		  m_axi_data_port_RDATA <= 'd1;
		  for(i=1;i<32;i=i+1) begin 
			  wait(m_axi_data_port_RVALID && m_axi_data_port_RREADY);
			  @(posedge ap_clk);
			  m_axi_data_port_RDATA <= i + 1; 
			  if(i==15) begin
			  m_axi_data_port_RLAST <= 1;
			  @(posedge ap_clk);
			  m_axi_data_port_RLAST <=0; end
			  else if(i==31) begin
			  m_axi_data_port_RLAST <= 1;
			  @(posedge ap_clk);
			  m_axi_data_port_RLAST <=0;
			  m_axi_data_port_RVALID <=0;end		  
			end
			
		wait(m_axi_data_port_RVALID && m_axi_data_port_RREADY);
		@(posedge ap_clk);
		m_axi_data_port_ARREADY <= 0;
		
		/* @(posedge ap_clk);
		m_axi_data_port_ARREADY <= 1;
		
		@(negedge m_axi_data_port_ARVALID);
		//@(posedge ap_clk);
		  m_axi_data_port_RVALID <= 1'b1;
		  m_axi_data_port_RDATA <= 'd1;
		  for(i=1;i<33;i=i+1) begin 
			  wait(m_axi_data_port_RVALID && m_axi_data_port_RREADY);
			  @(posedge ap_clk);
			  m_axi_data_port_RDATA <= i + 1; 
			  if(i==15) 
			  m_axi_data_port_RLAST <= 1;
			  @(posedge ap_clk);
			  m_axi_data_port_RLAST <=0;	  
		  end
			
		wait(m_axi_data_port_RVALID && m_axi_data_port_RREADY);
		m_axi_data_port_RVALID <=0;
		@(posedge ap_clk);
		m_axi_data_port_ARREADY <= 0;
		 */
		
	end
	
	// dep_queue verification
	initial begin

		l2g_dep_queue_V_TREADY = 0;
		g2l_dep_queue_V_TVALID = 0;
		g2l_dep_queue_V_TDATA = 0;
		
		repeat(6)
		@(posedge ap_clk);
		g2l_dep_queue_V_TDATA <= 1;	
		g2l_dep_queue_V_TVALID <= 1;
		
		@(posedge ap_clk);
		g2l_dep_queue_V_TVALID <= 0;
		
		repeat(60)
		@(posedge ap_clk);
		l2g_dep_queue_V_TREADY <= 1;
		//g2l_dep_queue_V_TDATA <= 0;
		@(posedge ap_clk);
		l2g_dep_queue_V_TREADY <= 0;
		
		@(posedge ap_clk);
		g2l_dep_queue_V_TVALID <= 1;
		@(posedge ap_clk);
		g2l_dep_queue_V_TVALID <= 0;
		
		repeat(82)
		@(posedge ap_clk);
		l2g_dep_queue_V_TREADY <= 1;
		//g2l_dep_queue_V_TDATA <= 0;
		@(posedge ap_clk);
		l2g_dep_queue_V_TREADY <= 0;
	end


load_1 u_load (
  .s_axi_CONTROL_BUS_AWADDR(s_axi_CONTROL_BUS_AWADDR),    // input wire [4 : 0] s_axi_CONTROL_BUS_AWADDR
  .s_axi_CONTROL_BUS_AWVALID(s_axi_CONTROL_BUS_AWVALID),  // input wire s_axi_CONTROL_BUS_AWVALID
  .s_axi_CONTROL_BUS_AWREADY(s_axi_CONTROL_BUS_AWREADY),  // output wire s_axi_CONTROL_BUS_AWREADY
  .s_axi_CONTROL_BUS_WDATA(s_axi_CONTROL_BUS_WDATA),      // input wire [31 : 0] s_axi_CONTROL_BUS_WDATA
  .s_axi_CONTROL_BUS_WSTRB(s_axi_CONTROL_BUS_WSTRB),      // input wire [3 : 0] s_axi_CONTROL_BUS_WSTRB
  .s_axi_CONTROL_BUS_WVALID(s_axi_CONTROL_BUS_WVALID),    // input wire s_axi_CONTROL_BUS_WVALID
  .s_axi_CONTROL_BUS_WREADY(s_axi_CONTROL_BUS_WREADY),    // output wire s_axi_CONTROL_BUS_WREADY
  .s_axi_CONTROL_BUS_BRESP(s_axi_CONTROL_BUS_BRESP),      // output wire [1 : 0] s_axi_CONTROL_BUS_BRESP
  .s_axi_CONTROL_BUS_BVALID(s_axi_CONTROL_BUS_BVALID),    // output wire s_axi_CONTROL_BUS_BVALID
  .s_axi_CONTROL_BUS_BREADY(s_axi_CONTROL_BUS_BREADY),    // input wire s_axi_CONTROL_BUS_BREADY
  .s_axi_CONTROL_BUS_ARADDR(s_axi_CONTROL_BUS_ARADDR),    // input wire [4 : 0] s_axi_CONTROL_BUS_ARADDR
  .s_axi_CONTROL_BUS_ARVALID(s_axi_CONTROL_BUS_ARVALID),  // input wire s_axi_CONTROL_BUS_ARVALID
  //.s_axi_CONTROL_BUS_ARREADY(s_axi_CONTROL_BUS_ARREADY),  // output wire s_axi_CONTROL_BUS_ARREADY
  //.s_axi_CONTROL_BUS_RDATA(s_axi_CONTROL_BUS_RDATA),      // output wire [31 : 0] s_axi_CONTROL_BUS_RDATA
  //.s_axi_CONTROL_BUS_RRESP(s_axi_CONTROL_BUS_RRESP),      // output wire [1 : 0] s_axi_CONTROL_BUS_RRESP
  //.s_axi_CONTROL_BUS_RVALID(s_axi_CONTROL_BUS_RVALID),    // output wire s_axi_CONTROL_BUS_RVALID
  .s_axi_CONTROL_BUS_RREADY(s_axi_CONTROL_BUS_RREADY),    // input wire s_axi_CONTROL_BUS_RREADY
  .ap_clk(ap_clk),                                        // input wire ap_clk
  .ap_rst_n(ap_rst_n),                                    // input wire ap_rst_n
  //.interrupt(),                                  // output wire interrupt
  .m_axi_data_port_AWADDR(m_axi_data_port_AWADDR),        // output wire [31 : 0] m_axi_data_port_AWADDR
  //.m_axi_data_port_AWLEN(m_axi_data_port_AWLEN),          // output wire [7 : 0] m_axi_data_port_AWLEN
  .m_axi_data_port_AWSIZE(m_axi_data_port_AWSIZE),        // output wire [2 : 0] m_axi_data_port_AWSIZE
  //.m_axi_data_port_AWBURST(m_axi_data_port_AWBURST),      // output wire [1 : 0] m_axi_data_port_AWBURST
  //.m_axi_data_port_AWLOCK(m_axi_data_port_AWLOCK),        // output wire [1 : 0] m_axi_data_port_AWLOCK
  //.m_axi_data_port_AWREGION(m_axi_data_port_AWREGION),    // output wire [3 : 0] m_axi_data_port_AWREGION
  //.m_axi_data_port_AWCACHE(m_axi_data_port_AWCACHE),      // output wire [3 : 0] m_axi_data_port_AWCACHE
  //.m_axi_data_port_AWPROT(m_axi_data_port_AWPROT),        // output wire [2 : 0] m_axi_data_port_AWPROT
  //.m_axi_data_port_AWQOS(m_axi_data_port_AWQOS),          // output wire [3 : 0] m_axi_data_port_AWQOS
  .m_axi_data_port_AWVALID(m_axi_data_port_AWVALID),      // output wire m_axi_data_port_AWVALID
  .m_axi_data_port_AWREADY(m_axi_data_port_AWREADY),      // input wire m_axi_data_port_AWREADY
  .m_axi_data_port_WDATA(m_axi_data_port_WDATA),          // output wire [63 : 0] m_axi_data_port_WDATA
  .m_axi_data_port_WSTRB(m_axi_data_port_WSTRB),          // output wire [7 : 0] m_axi_data_port_WSTRB
  .m_axi_data_port_WLAST(m_axi_data_port_WLAST),          // output wire m_axi_data_port_WLAST
  .m_axi_data_port_WVALID(m_axi_data_port_WVALID),        // output wire m_axi_data_port_WVALID
  .m_axi_data_port_WREADY(m_axi_data_port_WREADY),        // input wire m_axi_data_port_WREADY
  .m_axi_data_port_BRESP(m_axi_data_port_BRESP),          // input wire [1 : 0] m_axi_data_port_BRESP
  .m_axi_data_port_BVALID(m_axi_data_port_BVALID),        // input wire m_axi_data_port_BVALID
  .m_axi_data_port_BREADY(m_axi_data_port_BREADY),        // output wire m_axi_data_port_BREADY
  .m_axi_data_port_ARADDR(m_axi_data_port_ARADDR),        // output wire [31 : 0] m_axi_data_port_ARADDR
  //.m_axi_data_port_ARLEN(m_axi_data_port_ARLEN),          // output wire [7 : 0] m_axi_data_port_ARLEN
  //.m_axi_data_port_ARSIZE(m_axi_data_port_ARSIZE),        // output wire [2 : 0] m_axi_data_port_ARSIZE
  //.m_axi_data_port_ARBURST(m_axi_data_port_ARBURST),      // output wire [1 : 0] m_axi_data_port_ARBURST
  //.m_axi_data_port_ARLOCK(m_axi_data_port_ARLOCK),        // output wire [1 : 0] m_axi_data_port_ARLOCK
  //.m_axi_data_port_ARREGION(m_axi_data_port_ARREGION),    // output wire [3 : 0] m_axi_data_port_ARREGION
  //.m_axi_data_port_ARCACHE(m_axi_data_port_ARCACHE),      // output wire [3 : 0] m_axi_data_port_ARCACHE
  //.m_axi_data_port_ARPROT(m_axi_data_port_ARPROT),        // output wire [2 : 0] m_axi_data_port_ARPROT
  //.m_axi_data_port_ARQOS(m_axi_data_port_ARQOS),          // output wire [3 : 0] m_axi_data_port_ARQOS
  .m_axi_data_port_ARVALID(m_axi_data_port_ARVALID),      // output wire m_axi_data_port_ARVALID
  .m_axi_data_port_ARREADY(m_axi_data_port_ARREADY),      // input wire m_axi_data_port_ARREADY
  .m_axi_data_port_RDATA(m_axi_data_port_RDATA),          // input wire [63 : 0] m_axi_data_port_RDATA
  .m_axi_data_port_RRESP(m_axi_data_port_RRESP),          // input wire [1 : 0] m_axi_data_port_RRESP
  .m_axi_data_port_RLAST(m_axi_data_port_RLAST),          // input wire m_axi_data_port_RLAST
  .m_axi_data_port_RVALID(m_axi_data_port_RVALID),        // input wire m_axi_data_port_RVALID
  .m_axi_data_port_RREADY(m_axi_data_port_RREADY),        // output wire m_axi_data_port_RREADY
  .load_queue_V_V_TVALID(load_queue_V_V_TVALID),          // input wire load_queue_V_V_TVALID
  .load_queue_V_V_TREADY(load_queue_V_V_TREADY),          // output wire load_queue_V_V_TREADY
  .load_queue_V_V_TDATA(load_queue_V_V_TDATA),            // input wire [127 : 0] load_queue_V_V_TDATA
  .g2l_dep_queue_V_TVALID(g2l_dep_queue_V_TVALID),        // input wire g2l_dep_queue_V_TVALID
  .g2l_dep_queue_V_TREADY(g2l_dep_queue_V_TREADY),        // output wire g2l_dep_queue_V_TREADY
  .g2l_dep_queue_V_TDATA(g2l_dep_queue_V_TDATA),          // input wire [7 : 0] g2l_dep_queue_V_TDATA
  .l2g_dep_queue_V_TVALID(l2g_dep_queue_V_TVALID),        // output wire l2g_dep_queue_V_TVALID
  .l2g_dep_queue_V_TREADY(l2g_dep_queue_V_TREADY),        // input wire l2g_dep_queue_V_TREADY
  .l2g_dep_queue_V_TDATA(l2g_dep_queue_V_TDATA),          // output wire [7 : 0] l2g_dep_queue_V_TDATA
  //.inp_mem_V_Clk_A(inp_mem_V_Clk_A),                      // output wire inp_mem_V_Clk_A
  //.inp_mem_V_Rst_A(inp_mem_V_Rst_A),                      // output wire inp_mem_V_Rst_A
  //.inp_mem_V_EN_A(inp_mem_V_EN_A),                        // output wire inp_mem_V_EN_A
  //.inp_mem_V_WEN_A(inp_mem_V_WEN_A),                      // output wire [15 : 0] inp_mem_V_WEN_A
  //.inp_mem_V_Addr_A(inp_mem_V_Addr_A),                    // output wire [31 : 0] inp_mem_V_Addr_A
  //.inp_mem_V_Din_A(inp_mem_V_Din_A),                      // output wire [127 : 0] inp_mem_V_Din_A
  //.inp_mem_V_Dout_A(inp_mem_V_Dout_A),                    // input wire [127 : 0] inp_mem_V_Dout_A
  .wgt_mem_0_V_Clk_A(wgt_mem_0_V_Clk_A),                  // output wire wgt_mem_0_V_Clk_A
  .wgt_mem_0_V_Rst_A(wgt_mem_0_V_Rst_A),                  // output wire wgt_mem_0_V_Rst_A
  .wgt_mem_0_V_EN_A(wgt_mem_0_V_EN_A),                    // output wire wgt_mem_0_V_EN_A
  .wgt_mem_0_V_WEN_A(wgt_mem_0_V_WEN_A),                  // output wire [127 : 0] wgt_mem_0_V_WEN_A
  .wgt_mem_0_V_Addr_A(wgt_mem_0_V_Addr_A),                // output wire [31 : 0] wgt_mem_0_V_Addr_A
  .wgt_mem_0_V_Din_A(wgt_mem_0_V_Din_A),                  // output wire [1023 : 0] wgt_mem_0_V_Din_A
  .wgt_mem_0_V_Dout_A(wgt_mem_0_V_Dout_A),                // input wire [1023 : 0] wgt_mem_0_V_Dout_A
  .wgt_mem_1_V_Clk_A(wgt_mem_1_V_Clk_A),                  // output wire wgt_mem_1_V_Clk_A
  .wgt_mem_1_V_Rst_A(wgt_mem_1_V_Rst_A),                  // output wire wgt_mem_1_V_Rst_A
  .wgt_mem_1_V_EN_A(wgt_mem_1_V_EN_A),                    // output wire wgt_mem_1_V_EN_A
  .wgt_mem_1_V_WEN_A(wgt_mem_1_V_WEN_A),                  // output wire [127 : 0] wgt_mem_1_V_WEN_A
  .wgt_mem_1_V_Addr_A(wgt_mem_1_V_Addr_A),                // output wire [31 : 0] wgt_mem_1_V_Addr_A
  .wgt_mem_1_V_Din_A(wgt_mem_1_V_Din_A),                  // output wire [1023 : 0] wgt_mem_1_V_Din_A
  .wgt_mem_1_V_Dout_A(wgt_mem_1_V_Dout_A)                // input wire [1023 : 0] wgt_mem_1_V_Dout_A
);

	wire wgt_mem_0_V_Clk_A, wgt_mem_0_V_Rst_A, wgt_mem_0_V_EN_A;
	wire [127:0] wgt_mem_0_V_WEN_A;
	wire [31:0] wgt_mem_0_V_Addr_A;
	wire [1023:0] wgt_mem_0_V_Din_A;
	wire [1023:0] wgt_mem_0_V_Dout_A;


blk_mem_gen_0 u0 (
  .clka(wgt_mem_0_V_Clk_A),            // input wire clka
  .rsta(wgt_mem_0_V_Rst_A),            // input wire rsta
  .ena(wgt_mem_0_V_EN_A),              // input wire ena
  .wea(wgt_mem_0_V_WEN_A),              // input wire [127 : 0] wea
  .addra(wgt_mem_0_V_Addr_A),          // input wire [31 : 0] addra
  .dina(wgt_mem_0_V_Din_A),            // input wire [1023 : 0] dina
  .douta(wgt_mem_0_V_Dout_A)          // output wire [1023 : 0] douta
//  .clkb(clkb),            // input wire clkb
//  .rstb(rstb),            // input wire rstb
//  .enb(enb),              // input wire enb
//  .web(web),              // input wire [127 : 0] web
//  .addrb(addrb),          // input wire [31 : 0] addrb
//  .dinb(dinb),            // input wire [1023 : 0] dinb
//  .doutb(doutb),          // output wire [1023 : 0] doutb
//  .rsta_busy(rsta_busy),  // output wire rsta_busy
//  .rstb_busy(rstb_busy)  // output wire rstb_busy
);
	wire wgt_mem_1_V_Clk_A, wgt_mem_1_V_Rst_A, wgt_mem_1_V_EN_A;
	wire [127:0] wgt_mem_1_V_WEN_A;
	wire [31:0] wgt_mem_1_V_Addr_A;
	wire [1023:0] wgt_mem_1_V_Din_A;
	wire [1023:0] wgt_mem_1_V_Dout_A;


blk_mem_gen_0 u1 (
  .clka(wgt_mem_1_V_Clk_A),            // input wire clka
  .rsta(wgt_mem_1_V_Rst_A),            // input wire rsta
  .ena(wgt_mem_1_V_EN_A),              // input wire ena
  .wea(wgt_mem_1_V_WEN_A),              // input wire [127 : 0] wea
  .addra(wgt_mem_1_V_Addr_A),          // input wire [31 : 0] addra
  .dina(wgt_mem_1_V_Din_A),            // input wire [1023 : 0] dina
  .douta(wgt_mem_1_V_Dout_A)          // output wire [1023 : 0] douta
//  .clkb(clkb),            // input wire clkb
//  .rstb(rstb),            // input wire rstb
//  .enb(enb),              // input wire enb
//  .web(web),              // input wire [127 : 0] web
//  .addrb(addrb),          // input wire [31 : 0] addrb
//  .dinb(dinb),            // input wire [1023 : 0] dinb
//  .doutb(doutb),          // output wire [1023 : 0] doutb
//  .rsta_busy(rsta_busy),  // output wire rsta_busy
//  .rstb_busy(rstb_busy)  // output wire rstb_busy
);


endmodule
