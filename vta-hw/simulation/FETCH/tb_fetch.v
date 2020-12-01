`timescale 1ns/1ps
module tb_fetch();

	reg 		    ap_clk,ap_rst_n;
	reg	[4:0]  	s_axi_CONTROL_BUS_AWADDR;
	reg 	   	  s_axi_CONTROL_BUS_AWVALID;
	reg [31:0] 	s_axi_CONTROL_BUS_WDATA;
	reg [3:0]  	s_axi_CONTROL_BUS_WSTRB;
	reg 	   	  s_axi_CONTROL_BUS_WVALID;
	reg 	   	  s_axi_CONTROL_BUS_BREADY;
	reg 		    s_axi_CONTROL_BUS_BREADY;
  reg [4:0] 	s_axi_CONTROL_BUS_ARADDR;
  reg 		    s_axi_CONTROL_BUS_ARVALID;
	reg 		    s_axi_CONTROL_BUS_RREADY;
	reg			    m_axi_ins_port_AWREADY;
	reg 		    m_axi_ins_port_WREADY;
	reg [1:0] 	m_axi_ins_port_BRESP;
	reg 		    m_axi_ins_port_BVALID;
	reg 		    m_axi_ins_port_ARREADY;
  reg [127:0] m_axi_ins_port_RDATA;
  reg [1:0] 	m_axi_ins_port_RRESP;
	reg			    m_axi_ins_port_RVALID;
  reg 		    m_axi_ins_port_RLAST;
	reg 		    load_queue_V_V_TREADY;
	reg			    gemm_queue_V_V_TREADY;
	reg			    store_queue_V_V_TREADY;
	
	wire [127:0] load_insn;
	wire [127:0] store_insn;
	wire [127:0] gemm_insn;
	wire [127:0] alu_insn;
	
	parameter Clockperiod = 10;
	
	initial begin
		ap_clk = 0;
		forever #(Clockperiod/2) ap_clk = ~ap_clk; end

	task reset_task;
		input[15:0] reset_time;
		begin
			ap_rst_n = 0;
			#reset_time;
			ap_rst_n = 1; end
	endtask


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
	
	parameter VTA_LOG_UOP_BUFF_DEPTH = 13;
	parameter VTA_LOOP_ITER_WIDTH = 14;
	parameter VTA_LOG_ACC_BUFF_DEPTH = 11;
	parameter VTA_LOG_INP_BUFF_DEPTH = 11;
	parameter VTA_LOG_WGT_BUFF_DEPTH = 10;
	parameter VTA_ALU_OPCODE_BIT_WIDTH = 2;
	parameter VTA_ALUOP_IMM_BIT_WIDTH = 16;
	
	reg [OPCODE_BIT_WIDTH-1:0] opcode;
	reg [3:0] dep;
	reg [MEMOP_ID_BIT_WIDTH-1:0] buffer_id;
	reg [MEMOP_SRAM_ADDR_WIDTH-1:0] sram_base;
	reg [MEMOP_DRAM_ADDR_WIDTH-1:0] dram_base;
	reg [64-OPCODE_BIT_WIDTH-4-MEMOP_ID_BIT_WIDTH-MEMOP_SRAM_ADDR_WIDTH-MEMOP_DRAM_ADDR_WIDTH-1:0] unused_1;
	reg [MEMOP_SIZE_BIT_WIDTH-1:0] y_size, x_size;
	reg [MEMOP_STRIDE_BIT_WIDTH-1:0] x_stride;
	reg [MEMOP_PAD_BIT_WIDTH-1:0] y_pad_0, y_pad_1, x_pad_0, x_pad_1;
	reg RESET;
	reg [VTA_LOG_UOP_BUFF_DEPTH-1:0] uop_begin;
	reg [VTA_LOG_UOP_BUFF_DEPTH:0] uop_end;
	reg [VTA_LOOP_ITER_WIDTH-1:0] end0;
	reg [VTA_LOOP_ITER_WIDTH-1:0] end1;
	reg [64-OPCODE_BIT_WIDTH-4-1-VTA_LOG_UOP_BUFF_DEPTH*2-VTA_LOOP_ITER_WIDTH*2-1:0] unused_3;
	reg [VTA_LOG_ACC_BUFF_DEPTH-1:0] x0;
	reg [VTA_LOG_ACC_BUFF_DEPTH-1:0] x1;
	reg [VTA_LOG_INP_BUFF_DEPTH-1:0] y0;
	reg [VTA_LOG_INP_BUFF_DEPTH-1:0] y1;
	reg [VTA_LOG_WGT_BUFF_DEPTH-1:0] z0;
	reg [VTA_LOG_WGT_BUFF_DEPTH-1:0] z1;
	reg [64-VTA_LOG_ACC_BUFF_DEPTH*2-VTA_LOG_INP_BUFF_DEPTH*2-VTA_LOG_WGT_BUFF_DEPTH*2-1:0] unused_4; 
	reg [VTA_ALU_OPCODE_BIT_WIDTH-1:0] alu_op;
	reg USE_IMM;
	reg [VTA_ALUOP_IMM_BIT_WIDTH-1:0] imm;
	reg [64-VTA_LOG_ACC_BUFF_DEPTH*2-VTA_LOG_INP_BUFF_DEPTH*2-VTA_ALU_OPCODE_BIT_WIDTH-1-VTA_ALUOP_IMM_BIT_WIDTH-1:0] unused_5;
	
	// reg [64-2*MEMOP_SIZE_BIT_WIDTH-MEMOP_STRIDE_BIT_WIDTH-4*MEMOP_PAD_BIT_WIDTH-1:0] unused_2; 
	
	assign load_insn = {/*unused_2,*/ x_pad_1, x_pad_0, y_pad_1, y_pad_0, x_stride, x_size, y_size, unused_1, dram_base, sram_base, buffer_id, dep, opcode};
	assign store_insn = {/*unused_2,*/ x_pad_1, x_pad_0, y_pad_1, y_pad_0, x_stride, x_size, y_size, unused_1, dram_base, sram_base, buffer_id, dep, opcode};
	assign gemm_insn = {/* unused_4, */ z1, z0, y1, y0, x1, x0, unused_3, end1, end0, uop_end, uop_begin, RESET, dep, opcode};
	assign alu_insn = {unused_5, imm, USE_IMM, alu_op, y0, y0, x0, x0, unused_3, end1, end0, uop_end, uop_begin, RESET, dep, opcode};
	
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
		
		#235;
		opcode = OPCODE_STORE;
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
		
		@(posedge ap_clk);
		opcode = OPCODE_ALU;
		dep = 4'b0000;
		RESET = 0;
		uop_begin = 1;
		uop_end = 3;
		end0 = 1;
		end1 = 3;
		unused_3 = 0;
		x0 = 0;
		//x1 = 1;
		y0 = 0;
		//y1 = 1;
		alu_op = 1;
		USE_IMM = 1;
		imm = 1;
		unused_5 = 0;
		
		@(posedge ap_clk);
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
	
		@(posedge ap_clk);
		opcode = OPCODE_GEMM;
		dep = 4'b0000;
		RESET = 0;
		uop_begin = 1;
		uop_end = 3;
		end0 = 1;
		end1 = 3;
		unused_3 = 0;
		x0 = 0;
		x1 = 0;
		y0 = 0;
		y1 = 0;
		z0 = 'b1;
		z1 = 'b1; 
		unused_4 = 0;
	end

	initial begin 
		reset_task(30);
		s_axi_CONTROL_BUS_AWADDR = 0;
		s_axi_CONTROL_BUS_AWVALID = 0;	
		s_axi_CONTROL_BUS_WDATA = 0;
		s_axi_CONTROL_BUS_WSTRB = 'hf;
		s_axi_CONTROL_BUS_WVALID = 0;	
		s_axi_CONTROL_BUS_BREADY = 0;
		
		@(posedge ap_clk);
		s_axi_CONTROL_BUS_AWADDR <= 5'h10;
		s_axi_CONTROL_BUS_AWVALID <= 1'b1;
		
		@(posedge ap_clk);
		s_axi_CONTROL_BUS_AWVALID <= 1'b0;
		s_axi_CONTROL_BUS_WDATA <= 32'h5;
		s_axi_CONTROL_BUS_WVALID <= 1'b1;
		
		
		@(posedge ap_clk);
		s_axi_CONTROL_BUS_WVALID <= 1'b0;
		s_axi_CONTROL_BUS_BREADY <= 1;
		@(posedge ap_clk);
		s_axi_CONTROL_BUS_BREADY <= 0;
		
		//repeat(3)
		@(posedge ap_clk);
		s_axi_CONTROL_BUS_AWADDR <= 5'h18;
		s_axi_CONTROL_BUS_AWVALID <= 1'b1;
		
		@(posedge ap_clk);
		s_axi_CONTROL_BUS_AWVALID <= 1'b0;
		s_axi_CONTROL_BUS_WDATA <= 32'h11;
		s_axi_CONTROL_BUS_WVALID <= 1'b1;
		
		@(posedge ap_clk);
		s_axi_CONTROL_BUS_WVALID <= 1'b0;
		s_axi_CONTROL_BUS_BREADY <= 1;
		@(posedge ap_clk);
		s_axi_CONTROL_BUS_BREADY <= 0;
		
		//repeat(8)
		@(posedge ap_clk);
		s_axi_CONTROL_BUS_AWADDR <= 5'h0;
		s_axi_CONTROL_BUS_AWVALID <= 1'b1;
		
		@(posedge ap_clk);
		s_axi_CONTROL_BUS_AWVALID <= 1'b0;
		s_axi_CONTROL_BUS_WDATA <= 32'h81;
		s_axi_CONTROL_BUS_WVALID <= 1'b1;
		
		@(posedge ap_clk);
		s_axi_CONTROL_BUS_WVALID <= 1'b0;
		s_axi_CONTROL_BUS_BREADY <= 1;
		@(posedge ap_clk);
		s_axi_CONTROL_BUS_BREADY <= 0;
	end
	
	initial begin
		m_axi_ins_port_ARREADY = 0;
		m_axi_ins_port_RDATA = 0;
		m_axi_ins_port_RRESP = 0;
		m_axi_ins_port_RLAST = 0;
		m_axi_ins_port_RVALID = 0;
		
		#10;
		m_axi_ins_port_RDATA = load_insn;
		
		wait(m_axi_ins_port_ARVALID);
		m_axi_ins_port_ARREADY <= 1;
		
		@(negedge m_axi_ins_port_ARVALID);
		m_axi_ins_port_RVALID <= 1;
		m_axi_ins_port_ARREADY <= 0;
		@(posedge ap_clk);
		m_axi_ins_port_RDATA <= store_insn;
		
		@(posedge ap_clk);
		m_axi_ins_port_RDATA <= alu_insn;
		
		@(posedge ap_clk);
		m_axi_ins_port_RDATA <= load_insn;
		
		@(posedge ap_clk);
		m_axi_ins_port_RDATA <= gemm_insn;
		
		@(posedge ap_clk);		
		m_axi_ins_port_RLAST <= 1;
		
		@(posedge ap_clk);
		m_axi_ins_port_RLAST <= 0;
		m_axi_ins_port_RVALID <= 0;
		
		
	/* 	@(posedge m_axi_ins_port_ARVALID);
		//@(posedge ap_clk);
		m_axi_ins_port_ARREADY <= 1;
		//m_axi_ins_port_RDATA <= insn;
		
		@(negedge m_axi_ins_port_ARVALID);
		m_axi_ins_port_RVALID <= 1;
		m_axi_ins_port_ARREADY <= 0;
		
		repeat(3)
		@(posedge ap_clk);		
		m_axi_ins_port_RLAST <= 1;
		
		@(posedge ap_clk);
		m_axi_ins_port_RLAST <= 0;
		m_axi_ins_port_RVALID <= 0; */
		
	end

	initial begin
		load_queue_V_V_TREADY = 0;
		
		wait(load_queue_V_V_TVALID);
		load_queue_V_V_TREADY = 1;
		@(negedge load_queue_V_V_TVALID);
		load_queue_V_V_TREADY <= 0;
	end
	
	initial begin
		store_queue_V_V_TREADY = 0;
		
		wait(store_queue_V_V_TVALID);
		store_queue_V_V_TREADY = 1;
		@(negedge store_queue_V_V_TVALID);
		store_queue_V_V_TREADY <= 0;
		
	end
	
	initial begin
		gemm_queue_V_V_TREADY = 0;
		
		wait(gemm_queue_V_V_TVALID);
		gemm_queue_V_V_TREADY = 1;
		@(negedge gemm_queue_V_V_TVALID);
		gemm_queue_V_V_TREADY <= 0;
	
		wait(gemm_queue_V_V_TVALID);
		gemm_queue_V_V_TREADY = 1;
		@(negedge gemm_queue_V_V_TVALID);
		gemm_queue_V_V_TREADY <= 0;
	end


fetch_0  tb (
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
  .s_axi_CONTROL_BUS_ARREADY(s_axi_CONTROL_BUS_ARREADY),  // output wire s_axi_CONTROL_BUS_ARREADY
  .s_axi_CONTROL_BUS_RDATA(s_axi_CONTROL_BUS_RDATA),      // output wire [31 : 0] s_axi_CONTROL_BUS_RDATA
  .s_axi_CONTROL_BUS_RRESP(s_axi_CONTROL_BUS_RRESP),      // output wire [1 : 0] s_axi_CONTROL_BUS_RRESP
  .s_axi_CONTROL_BUS_RVALID(s_axi_CONTROL_BUS_RVALID),    // output wire s_axi_CONTROL_BUS_RVALID
  .s_axi_CONTROL_BUS_RREADY(s_axi_CONTROL_BUS_RREADY),    // input wire s_axi_CONTROL_BUS_RREADY
  .ap_clk(ap_clk),                                        // input wire ap_clk
  .ap_rst_n(ap_rst_n),                                    // input wire ap_rst_n
  .interrupt(interrupt),                                  // output wire interrupt
  .m_axi_ins_port_AWADDR(m_axi_ins_port_AWADDR),          // output wire [31 : 0] m_axi_ins_port_AWADDR
  .m_axi_ins_port_AWLEN(m_axi_ins_port_AWLEN),            // output wire [7 : 0] m_axi_ins_port_AWLEN
  .m_axi_ins_port_AWSIZE(m_axi_ins_port_AWSIZE),          // output wire [2 : 0] m_axi_ins_port_AWSIZE
  .m_axi_ins_port_AWBURST(m_axi_ins_port_AWBURST),        // output wire [1 : 0] m_axi_ins_port_AWBURST
  .m_axi_ins_port_AWLOCK(m_axi_ins_port_AWLOCK),          // output wire [1 : 0] m_axi_ins_port_AWLOCK
  .m_axi_ins_port_AWREGION(m_axi_ins_port_AWREGION),      // output wire [3 : 0] m_axi_ins_port_AWREGION
  .m_axi_ins_port_AWCACHE(m_axi_ins_port_AWCACHE),        // output wire [3 : 0] m_axi_ins_port_AWCACHE
  .m_axi_ins_port_AWPROT(m_axi_ins_port_AWPROT),          // output wire [2 : 0] m_axi_ins_port_AWPROT
  .m_axi_ins_port_AWQOS(m_axi_ins_port_AWQOS),            // output wire [3 : 0] m_axi_ins_port_AWQOS
  .m_axi_ins_port_AWVALID(m_axi_ins_port_AWVALID),        // output wire m_axi_ins_port_AWVALID
  .m_axi_ins_port_AWREADY(m_axi_ins_port_AWREADY),        // input wire m_axi_ins_port_AWREADY
  .m_axi_ins_port_WDATA(m_axi_ins_port_WDATA),            // output wire [127 : 0] m_axi_ins_port_WDATA
  .m_axi_ins_port_WSTRB(m_axi_ins_port_WSTRB),            // output wire [15 : 0] m_axi_ins_port_WSTRB
  .m_axi_ins_port_WLAST(m_axi_ins_port_WLAST),            // output wire m_axi_ins_port_WLAST
  .m_axi_ins_port_WVALID(m_axi_ins_port_WVALID),          // output wire m_axi_ins_port_WVALID
  .m_axi_ins_port_WREADY(m_axi_ins_port_WREADY),          // input wire m_axi_ins_port_WREADY
  .m_axi_ins_port_BRESP(m_axi_ins_port_BRESP),            // input wire [1 : 0] m_axi_ins_port_BRESP
  .m_axi_ins_port_BVALID(m_axi_ins_port_BVALID),          // input wire m_axi_ins_port_BVALID
  .m_axi_ins_port_BREADY(m_axi_ins_port_BREADY),          // output wire m_axi_ins_port_BREADY
  .m_axi_ins_port_ARADDR(m_axi_ins_port_ARADDR),          // output wire [31 : 0] m_axi_ins_port_ARADDR
  .m_axi_ins_port_ARLEN(m_axi_ins_port_ARLEN),            // output wire [7 : 0] m_axi_ins_port_ARLEN
  .m_axi_ins_port_ARSIZE(m_axi_ins_port_ARSIZE),          // output wire [2 : 0] m_axi_ins_port_ARSIZE
  .m_axi_ins_port_ARBURST(m_axi_ins_port_ARBURST),        // output wire [1 : 0] m_axi_ins_port_ARBURST
  .m_axi_ins_port_ARLOCK(m_axi_ins_port_ARLOCK),          // output wire [1 : 0] m_axi_ins_port_ARLOCK
  .m_axi_ins_port_ARREGION(m_axi_ins_port_ARREGION),      // output wire [3 : 0] m_axi_ins_port_ARREGION
  .m_axi_ins_port_ARCACHE(m_axi_ins_port_ARCACHE),        // output wire [3 : 0] m_axi_ins_port_ARCACHE
  .m_axi_ins_port_ARPROT(m_axi_ins_port_ARPROT),          // output wire [2 : 0] m_axi_ins_port_ARPROT
  .m_axi_ins_port_ARQOS(m_axi_ins_port_ARQOS),            // output wire [3 : 0] m_axi_ins_port_ARQOS
  .m_axi_ins_port_ARVALID(m_axi_ins_port_ARVALID),        // output wire m_axi_ins_port_ARVALID
  .m_axi_ins_port_ARREADY(m_axi_ins_port_ARREADY),        // input wire m_axi_ins_port_ARREADY
  .m_axi_ins_port_RDATA(m_axi_ins_port_RDATA),            // input wire [127 : 0] m_axi_ins_port_RDATA
  .m_axi_ins_port_RRESP(m_axi_ins_port_RRESP),            // input wire [1 : 0] m_axi_ins_port_RRESP
  .m_axi_ins_port_RLAST(m_axi_ins_port_RLAST),            // input wire m_axi_ins_port_RLAST
  .m_axi_ins_port_RVALID(m_axi_ins_port_RVALID),          // input wire m_axi_ins_port_RVALID
  .m_axi_ins_port_RREADY(m_axi_ins_port_RREADY),          // output wire m_axi_ins_port_RREADY
  .load_queue_V_V_TVALID(load_queue_V_V_TVALID),          // output wire load_queue_V_V_TVALID
  .load_queue_V_V_TREADY(load_queue_V_V_TREADY),          // input wire load_queue_V_V_TREADY
  .load_queue_V_V_TDATA(load_queue_V_V_TDATA),            // output wire [127 : 0] load_queue_V_V_TDATA
  .gemm_queue_V_V_TVALID(gemm_queue_V_V_TVALID),          // output wire gemm_queue_V_V_TVALID
  .gemm_queue_V_V_TREADY(gemm_queue_V_V_TREADY),          // input wire gemm_queue_V_V_TREADY
  .gemm_queue_V_V_TDATA(gemm_queue_V_V_TDATA),            // output wire [127 : 0] gemm_queue_V_V_TDATA
  .store_queue_V_V_TVALID(store_queue_V_V_TVALID),        // output wire store_queue_V_V_TVALID
  .store_queue_V_V_TREADY(store_queue_V_V_TREADY),        // input wire store_queue_V_V_TREADY
  .store_queue_V_V_TDATA(store_queue_V_V_TDATA)          // output wire [127 : 0] store_queue_V_V_TDATA
);

endmodule
