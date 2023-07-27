#include "pim_blas.h"
#include <math.h>

#define Q_ADDR_OFFSET 	0x0
#define K_ADDR_OFFSET	0x20000
#define V_ADDR_OFFSET	0x40000

#define Q_META_OFFSET	0
#define K_META_OFFSET	69001
#define V_META_OFFSET	69106

#define QK_ADDR_OFFSET	0x401
#define QK_META_OFFSET	69211

#define SOFTMAX_ADDR_OFFSET	0x401
#define SOFTMAX_META_OFFSET	69316

#define SV_ADDR_OFFSET	0x401
#define SV_META_OFFSET	69421

uint8_t *null_ptr = (uint8_t *)malloc(sizeof(uint8_t) * WORD_SIZE);

void blas_init(uint64_t num) {
	struct PimCgInitData pim_hbm_init;

	pim_hbm_init.pim_reg[IDX_SBMR] = MAP_SBMR << ro_pos;
	pim_hbm_init.pim_reg[IDX_ABMR] = MAP_ABMR << ro_pos;
	pim_hbm_init.pim_reg[IDX_PIM_OP_MODE] = MAP_PIM_OP_MODE << ro_pos;
	pim_hbm_init.pim_reg[IDX_CRF] = MAP_CRF << ro_pos;
	pim_hbm_init.pim_reg[IDX_GRF] = MAP_GRF << ro_pos;
	pim_hbm_init.pim_reg[IDX_SRF] = MAP_SRF << ro_pos;

	pim_hbm_init.pim_op[0] = 0; // RD
	pim_hbm_init.pim_op[1] = 1; // WR

	CmdGenInit(pim_hbm_init);
}

bool C_pimblasAddPreprocess(int len, uint8_t **in0, uint8_t **in1) {
	return true;
}

bool C_pim_add(int len, uint8_t *in0, uint8_t *in1, uint8_t *out) {
	return true;	
}

bool C_pimblasMulPreprocess(int len, uint8_t **in0, uint8_t **in1) {
	return true;
}

bool C_pim_mul(int len, uint8_t *in0, uint8_t *in1, uint8_t *out) {
	return true;	
}

bool C_pimblasGemvPreprocess(int len_in, int len_out, uint8_t **w) {
	return true;
}

// Conv1D for transformer
bool C_pimblasConv1DPreprocess(){

}

// maskedMM for tarnsformer
bool C_pimblasMaskedMMPreprocess(){

}

// softmax for transformer
bool C_pimblasSoftmaxPreprocess(){

}

// layer normalization for transformer
bool C_pimblasLayerNormPreprocess(){

}

bool C_pim_gemv(int len_in, int len_out, uint8_t *in, uint8_t *w, uint8_t *out) {
	return true;
}

bool pimblasAddPreprocess(struct PimCmdMetadata *add_cmd_metadata, int len, uint8_t **in0, uint8_t **in1) {
	int in0_idx = 0;  // mem
	int in1_idx = 1;  // mem
	int out_idx = 2;  // mem
	int in0_addr = 0;
	int in1_addr = in0_addr + len / UNITS_PER_WORD;
	int out_addr = in1_addr + len / UNITS_PER_WORD;

	// Write Input to Memory
	// memAccess(in0, in0_addr, true, (uint32_t)(len * UNIT_SIZE));
	// memAccess(in1, in1_addr, true, (uint32_t)(len * UNIT_SIZE));

	add_cmd_metadata->n_iter = len * UNIT_SIZE / GRF_SIZE;
	add_cmd_metadata->n_cmdgroup = 4;
	add_cmd_metadata->operand[in0_idx] = in0_addr;
	add_cmd_metadata->operand[in1_idx] = in1_addr;

	// Read SB → AB (Automatic in FPGA, ok)

	// Write ukernel code in Metadata (Only Send Data to FPGA)
	add_cmd_metadata->code[0] = 0b01000010000000001000000000000000; // MOV(A)  GRF_A[A0]  BANK
	add_cmd_metadata->code[1] = 0b00010000000001000000100000000111; // JUMP    -1         7
	add_cmd_metadata->code[2] = 0b10000010000010001000000000000000; // ADD(A)  GRF_A[A0]  BANK      GRF_A[A0]
	add_cmd_metadata->code[3] = 0b00010000000001000000100000000111; // JUMP    -1         7
	add_cmd_metadata->code[4] = 0b01000000010000001000000000000000; // MOV(A)  BANK       GRF_A[A0]
	add_cmd_metadata->code[5] = 0b00010000000001000000100000000111; // JUMP    -1         7
	add_cmd_metadata->code[6] = 0b00100000000000000000000000000000; // EXIT	

	// Change to AB-PIM Mode
	add_cmd_metadata->cmd_info[0].op_code_tg = 0;  // RD
	add_cmd_metadata->cmd_info[0].addr_tg = IDX_PIM_OP_MODE;
	add_cmd_metadata->cmd_info[0].data_tg = 0;
	add_cmd_metadata->cmd_info[0].addr_tg_step = 0;
	add_cmd_metadata->cmd_info[0].data_tg_step = 0;
	add_cmd_metadata->cmd_info[0].n_cmd = 1;

	// Read Input0 8 Col
	add_cmd_metadata->cmd_info[1].op_code_tg = 0;  // RD
	add_cmd_metadata->cmd_info[1].addr_tg = in0_idx;
	add_cmd_metadata->cmd_info[1].data_tg = 0;
	add_cmd_metadata->cmd_info[1].addr_tg_step = 1 << co_pos;
	add_cmd_metadata->cmd_info[1].data_tg_step = 0;
	add_cmd_metadata->cmd_info[1].n_cmd = 8;

	// Read Input1 8 Col
	add_cmd_metadata->cmd_info[2].op_code_tg = 0;  // RD
	add_cmd_metadata->cmd_info[2].addr_tg = in1_idx;
	add_cmd_metadata->cmd_info[2].data_tg = 0;
	add_cmd_metadata->cmd_info[2].addr_tg_step = 1 << co_pos;
	add_cmd_metadata->cmd_info[2].data_tg_step = 0;
	add_cmd_metadata->cmd_info[2].n_cmd = 8;

	// WRITE Output 8 Col
	add_cmd_metadata->cmd_info[3].op_code_tg = 0;  // WR
	add_cmd_metadata->cmd_info[3].addr_tg = out_idx;
	add_cmd_metadata->cmd_info[3].data_tg = 0;
	add_cmd_metadata->cmd_info[3].addr_tg_step = 1 << co_pos;
	add_cmd_metadata->cmd_info[3].data_tg_step = 0;
	add_cmd_metadata->cmd_info[3].n_cmd = 8;
	
	return true;
}

bool pimBlasFFNPreprocess(uint32_t *packed_data, uint16_t *W, int seq_len, int in_dim){

	int weight_size = in_dim * in_dim * 4;

	for(int i = 0; i < 100000; i++)
		packed_data[i] = 0;

	uint32_t tmp_input[68896];
	for(int i = 0; i < 68896; i++)
		tmp_input[i] = i;

	// ffn
	packed_data[0] = 0x00010001; 							 // n_iter || n_cmd_group
	packed_data[1] = 0b00000111111111110110000000000000;	 // grf
	packed_data[2] = 0b00000111111111110100000000000000;	 // srf
	packed_data[3] = 0b00000111111111111010000000000000;	 // pim mode
	packed_data[4] = 0;		 	 							 // Wq x 8
	packed_data[5] = 0b00000111111111110100000000000000;	 // srf
	packed_data[6] = 0b00000111111111110110000000000000;	 // grf
	packed_data[7] = 0;	 		 							 // Q
	packed_data[69320] = 0x0;	 							 // EXIT

	packed_data[37] = 0x00010203; // addr tg
	packed_data[38] = 0x04050607; // addr tg

	packed_data[41] = 0x00000000; // data tg

	packed_data[45] = 0x01010020;	// addr tg step
	packed_data[46] = 0x01002000;	// addr tg step

	packed_data[53] = 0x01010101;	// data tg step
	packed_data[54] = 0x01010101;	// data tg step

	packed_data[61] = 0x00010001;	// n_cmd
	packed_data[62] = 0x00010008;	// n_cmd
	packed_data[63] = 0x00010007;	// n_cmd
	packed_data[64] = 0x00010001;	// n_cmd
	packed_data[65] = 0x00010001;	// n_cmd
	packed_data[66] = 0x00010001;	// n_cmd
	packed_data[67] = 0x00010001;	// n_cmd
	packed_data[68] = 0x00010001;	// n_cmd

	packed_data[69] = 0b10110100110000001000000000000000;	// u_kernel - MAC
	packed_data[70] = 0b10110100110000001000000000000000;	// u_kernel - MAC
	packed_data[71] = 0b10110100110000001000000000000000;	// u_kernel - MAC
	packed_data[72] = 0b10110100110000001000000000000000;	// u_kernel - MAC
	packed_data[73] = 0b10110100110000001000000000000000;	// u_kernel - MAC
	packed_data[74] = 0b10110100110000001000000000000000;	// u_kernel - MAC
	packed_data[75] = 0b10110100110000001000000000000000;	// u_kernel - MAC
	packed_data[76] = 0b10110100110000001000000000000000;	// u_kernel - MAC
	packed_data[77] = 0b10000100100100000000000000000111;	// u_kernel - ADD
	packed_data[78] = 0b10000100100100000000000000000110;	// u_kernel - ADD
	packed_data[79] = 0b10000100100100000000000000000101;	// u_kernel - ADD
	packed_data[80] = 0b10000100100100000000000000000100;	// u_kernel - ADD
	packed_data[81] = 0b10000100100100000000000000000011;	// u_kernel - ADD
	packed_data[82] = 0b10000100100100000000000000000010;	// u_kernel - ADD
	packed_data[83] = 0b10000100100100000000000000000001;	// u_kernel - ADD	
	packed_data[84] = 0b01000000100000000000000000000000;	// u_kernel - MOV
	packed_data[85] = 0b00010000000001000101000000011111;	// u_kernel - JUMP
	packed_data[86] = 0b00100000000000000000000000000000;	// u_kernel - EXIT

	packed_data[102] = 512; // dim size - from pytorch
	packed_data[103] = 1;	// num metadata 
	packed_data[104] = 1;	// execution

	// weight pre-load
	memAccess(W, V_ADDR_OFFSET, 1, sizeof(uint8_t) * weight_size);
}

void generateQKVMeta(uint32_t *packed_data, uint32_t addr_offset, uint32_t meta_offset,
					int seq_len, int in_dim){

	int len_in_ = Ceiling(in_dim, 8);
	int len_out_ = Ceiling(in_dim, 4096);
	
	int w_idx = 0;   // mem
	int out_idx = 1; // mem
	int in_idx = 0;  // data

	int w_addr = 0;
	int out_addr = w_addr + len_in_ * len_out_ / UNITS_PER_WORD;
	int in_addr = 0;

	// packed_data[meta_offset + 0] = (len_in_ + 8 - 1) / 8 || 3;			 // n_iter || n_cmd_group
	packed_data[meta_offset + 0] = 0x00010001;			 				 // n_iter || n_cmd_group
	packed_data[meta_offset + 1] = 0b00000111111111110110000000000000;	 // grf
	packed_data[meta_offset + 2] = 0b00000111111111110100000000000000;	 // srf
	packed_data[meta_offset + 3] = 0b00000111111111111010000000000000;	 // pim mode
	packed_data[meta_offset + 4] = addr_offset;		 	 				 // Wq x 8
	packed_data[meta_offset + 5] = 0b00000111111111110100000000000000;	 // srf
	packed_data[meta_offset + 6] = 0b00000111111111110110000000000000;	 // grf
	packed_data[meta_offset + 7] = addr_offset;	 		 				 // Q
	packed_data[meta_offset + 8] = 0x0;	 								 // EXIT

	packed_data[meta_offset + 37] = 0x00010203; // addr tg
	packed_data[meta_offset + 38] = 0x04050607; // addr tg

	packed_data[meta_offset + 41] = 0x00000000; // data tg

	packed_data[meta_offset + 45] = 0x01010020;	// addr tg step
	packed_data[meta_offset + 46] = 0x01002000;	// addr tg step

	packed_data[meta_offset + 53] = 0x01010101;	// data tg step
	packed_data[meta_offset + 54] = 0x01010101;	// data tg step

	packed_data[meta_offset + 61] = 0x00010001;	// n_cmd
	packed_data[meta_offset + 62] = 0x00010008;	// n_cmd
	packed_data[meta_offset + 63] = 0x00010007;	// n_cmd
	packed_data[meta_offset + 64] = 0x00010001;	// n_cmd
	packed_data[meta_offset + 65] = 0x00010001;	// n_cmd
	packed_data[meta_offset + 66] = 0x00010001;	// n_cmd
	packed_data[meta_offset + 67] = 0x00010001;	// n_cmd
	packed_data[meta_offset + 68] = 0x00010001;	// n_cmd

	packed_data[meta_offset + 69] = 0b10110100110000001000000000000000;	// u_kernel - MAC
	packed_data[meta_offset + 70] = 0b10110100110000001000000000000000;	// u_kernel - MAC
	packed_data[meta_offset + 71] = 0b10110100110000001000000000000000;	// u_kernel - MAC
	packed_data[meta_offset + 72] = 0b10110100110000001000000000000000;	// u_kernel - MAC
	packed_data[meta_offset + 73] = 0b10110100110000001000000000000000;	// u_kernel - MAC
	packed_data[meta_offset + 74] = 0b10110100110000001000000000000000;	// u_kernel - MAC
	packed_data[meta_offset + 75] = 0b10110100110000001000000000000000;	// u_kernel - MAC
	packed_data[meta_offset + 76] = 0b10110100110000001000000000000000;	// u_kernel - MAC
	packed_data[meta_offset + 77] = 0b10000100100100000000000000000111;	// u_kernel - ADD
	packed_data[meta_offset + 78] = 0b10000100100100000000000000000110;	// u_kernel - ADD
	packed_data[meta_offset + 79] = 0b10000100100100000000000000000101;	// u_kernel - ADD
	packed_data[meta_offset + 80] = 0b10000100100100000000000000000100;	// u_kernel - ADD
	packed_data[meta_offset + 81] = 0b10000100100100000000000000000011;	// u_kernel - ADD
	packed_data[meta_offset + 82] = 0b10000100100100000000000000000010;	// u_kernel - ADD
	packed_data[meta_offset + 83] = 0b10000100100100000000000000000001;	// u_kernel - ADD	
	packed_data[meta_offset + 84] = 0b01000000100000000000000000000000;	// u_kernel - MOV
	packed_data[meta_offset + 85] = 0b00010000000001000101000000011111;	// u_kernel - JUMP
	packed_data[meta_offset + 86] = 0b00100000000000000000000000000000;	// u_kernel - EXIT

	packed_data[meta_offset + 102] = in_dim; // dim size - from pytorch
	packed_data[meta_offset + 103] = 6;	// num metadata 
	packed_data[meta_offset + 104] = 1;	// execution
}

void generateMulQKMeta(uint32_t *packed_data, uint32_t addr_offset, uint32_t meta_offset,
					int seq_len, int in_dim){
	packed_data[meta_offset + 0] = 0x00010001; 							 // n_iter || n_cmd_group
	packed_data[meta_offset + 1] = 0b00000111111111111010000000000000;	 // pim mode
	packed_data[meta_offset + 2] = 0x0;	 								 // Q
	packed_data[meta_offset + 3] = meta_offset;							 // Score
	packed_data[meta_offset + 4] = 0x0;	 	 							 // Q
	packed_data[meta_offset + 5] = 0x0;	 	 							 // EXIT

	packed_data[meta_offset + 37] = 0x00010203; // addr tg
	packed_data[meta_offset + 38] = 0x04050607; // addr tg

	packed_data[meta_offset + 41] = 0x00000000; // data tg

	packed_data[meta_offset + 45] = 0x00202000;	// addr tg step
	packed_data[meta_offset + 46] = 0x01010101;	// addr tg step

	packed_data[meta_offset + 53] = 0x01010101;	// data tg step
	packed_data[meta_offset + 54] = 0x01010101;	// data tg step

	packed_data[meta_offset + 61] = 0x00010008;	// n_cmd
	packed_data[meta_offset + 62] = 0x00010001;	// n_cmd
	packed_data[meta_offset + 63] = 0x00010008;	// n_cmd
	packed_data[meta_offset + 64] = 0x00010001;	// n_cmd
	packed_data[meta_offset + 65] = 0x00010008;	// n_cmd
	packed_data[meta_offset + 66] = 0x00010001;	// n_cmd
	packed_data[meta_offset + 67] = 0x00010008;	// n_cmd
	packed_data[meta_offset + 68] = 0x00010001;	// n_cmd

	packed_data[meta_offset + 69] = 0b10010100100000001000000000000000;	// u_kernel - MUL
	packed_data[meta_offset + 70] = 0b01000000100000001000000000000000;	// u_kernel - MOV
	packed_data[meta_offset + 71] = 0b10010100100000001000000000000000;	// u_kernel - MUL
	packed_data[meta_offset + 72] = 0b01000000100000001000000000000000;	// u_kernel - MOV
	packed_data[meta_offset + 73] = 0b10010100100000001000000000000000;	// u_kernel - MUL
	packed_data[meta_offset + 74] = 0b01000000100000001000000000000000;	// u_kernel - MOV
	packed_data[meta_offset + 75] = 0b10010100100000001000000000000000;	// u_kernel - MUL
	packed_data[meta_offset + 76] = 0b01000000100000001000000000000000;	// u_kernel - MOV
	packed_data[meta_offset + 77] = 0b01000000000000000000100000000000;	// u_kernel - MOV
	packed_data[meta_offset + 78] = 0b00010000000001000100100011111111;	// u_kernel - JUMP -9 255
	packed_data[meta_offset + 79] = 0b00100000000000000000000000000000;	// u_kernel - EXIT
}

void generateSoftmaxMeta(uint32_t *packed_data, uint32_t addr_offset, uint32_t meta_offset,
					int seq_len, int in_dim){
	packed_data[meta_offset + 0] = 0x00010001; 							 // n_iter || n_cmd_group
	packed_data[meta_offset + 1] = 0b00000111111111111010000000000000;	 // pim mode
	packed_data[meta_offset + 2] = addr_offset;	 	 					 // Score
	packed_data[meta_offset + 3] = addr_offset;	 	 					 // Score
	packed_data[meta_offset + 4] = addr_offset;	 						 // Score
	packed_data[meta_offset + 5] = 0x0;	 								 // EXIT

	packed_data[meta_offset + 37] = 0x00010203; // addr tg
	packed_data[meta_offset + 38] = 0x04050607; // addr tg

	packed_data[meta_offset + 41] = 0x00000000; // data tg

	packed_data[meta_offset + 45] = 0x00202020;	// addr tg step
	packed_data[meta_offset + 46] = 0x20202001;	// addr tg step

	packed_data[meta_offset + 53] = 0x01010101;	// data tg step
	packed_data[meta_offset + 54] = 0x01010101;	// data tg step

	packed_data[meta_offset + 61] = 0x000100a0;	// n_cmd
	packed_data[meta_offset + 62] = 0x00a000a0;	// n_cmd
	packed_data[meta_offset + 63] = 0x00a000a0;	// n_cmd
	packed_data[meta_offset + 64] = 0x00010001;	// n_cmd
	packed_data[meta_offset + 65] = 0x00010001;	// n_cmd
	packed_data[meta_offset + 66] = 0x00010001;	// n_cmd
	packed_data[meta_offset + 67] = 0x00010001;	// n_cmd
	packed_data[meta_offset + 68] = 0x00010001;	// n_cmd

	packed_data[meta_offset + 69] = 0b11000000000000000000000000000000;	// u_kernel - EXP
	packed_data[meta_offset + 70] = 0b00010000000001000000100000001111;	// u_kernel - JUMP -1 15
	packed_data[meta_offset + 71] = 0b10100100000000001000000000000000;	// u_kernel - ADD
	packed_data[meta_offset + 72] = 0b00010000000001000000100000001111;	// u_kernel - JUMP -1 15
	packed_data[meta_offset + 73] = 0b11011000100000000000000000000000;	// u_kernel - REDUCE
	packed_data[meta_offset + 74] = 0b11100000001000000000000000000000;	// u_kernel - DIV
	packed_data[meta_offset + 75] = 0b00010000000001000000100000001111;	// u_kernel - JUMP -1 15
	packed_data[meta_offset + 76] = 0b00100000000000000000000000000000;	// u_kernel - EXIT
}

void generateMulSVMeta(uint32_t *packed_data, uint32_t addr_offset, uint32_t meta_offset,
					int seq_len, int in_dim){
	packed_data[meta_offset + 0] = 0x00010001; 							 // n_iter || n_cmd_group
	packed_data[meta_offset + 1] = 0b00000111111111111010000000000000;	 // pim mode
	packed_data[meta_offset + 2] = 0x0;	 	 							 // Q
	packed_data[meta_offset + 3] = addr_offset;	 						 // Score
	packed_data[meta_offset + 4] = 0x0;	 								 // Q
	packed_data[meta_offset + 5] = 0x0;	 								 // EXIT

	packed_data[meta_offset + 37] = 0x00010203; // addr tg
	packed_data[meta_offset + 38] = 0x04050607; // addr tg

	packed_data[meta_offset + 41] = 0x00000000; // data tg

	packed_data[meta_offset + 45] = 0x00202000;	// addr tg step
	packed_data[meta_offset + 46] = 0x01010101;	// addr tg step

	packed_data[meta_offset + 53] = 0x01010101;	// data tg step
	packed_data[meta_offset + 54] = 0x01010101;	// data tg step

	packed_data[meta_offset + 61] = 0x00010008;	// n_cmd
	packed_data[meta_offset + 62] = 0x00010001;	// n_cmd
	packed_data[meta_offset + 63] = 0x00010008;	// n_cmd
	packed_data[meta_offset + 64] = 0x00010001;	// n_cmd
	packed_data[meta_offset + 65] = 0x00010008;	// n_cmd
	packed_data[meta_offset + 66] = 0x00010001;	// n_cmd
	packed_data[meta_offset + 67] = 0x00010008;	// n_cmd
	packed_data[meta_offset + 68] = 0x00010001;	// n_cmd

	packed_data[meta_offset + 69] = 0b10010100100000001000000000000000;	// u_kernel - MUL
	packed_data[meta_offset + 70] = 0b01000000100000001000000000000000;	// u_kernel - MOV
	packed_data[meta_offset + 71] = 0b10010100100000001000000000000000;	// u_kernel - MUL
	packed_data[meta_offset + 72] = 0b01000000100000001000000000000000;	// u_kernel - MOV
	packed_data[meta_offset + 73] = 0b10010100100000001000000000000000;	// u_kernel - MUL
	packed_data[meta_offset + 74] = 0b01000000100000001000000000000000;	// u_kernel - MOV
	packed_data[meta_offset + 75] = 0b10010100100000001000000000000000;	// u_kernel - MUL
	packed_data[meta_offset + 76] = 0b01000000100000001000000000000000;	// u_kernel - MOV
	packed_data[meta_offset + 77] = 0b01000000000000000000100000000000;	// u_kernel - MOV
	packed_data[meta_offset + 78] = 0b00010000000001000100100011111111;	// u_kernel - JUMP -9 255
	packed_data[meta_offset + 79] = 0b00100000000000000000000000000000;	// u_kernel - EXIT
}		

// bool pimBlasAttentionPreprocess(int seq_len, int in_dim, uint32_t *in_token, uint32_t *packed_data){
bool pimBlasAttentionPreprocess(int seq_len, int in_dim, uint16_t *in_token, 
								uint16_t *W, uint32_t *packed_data){

	// int len_in_ = Ceiling(len_in, 8);
	// int len_out_ = Ceiling(len_out, 4096);

	// int w_idx = 0;		// mem
	// int out_idx = 1;	// mem
	// int in_idx = 0;		// data

	// int w_addr = 0;
	// int out_addr = w_addr + len_in_ * len_out_ / UNIT_PER_WORD;
	// int in_addr = 0;

	int weight_size = in_dim * in_dim * 4;

	for(int i = 0; i < 100000; i++)
		packed_data[i] = 0;

	uint32_t tmp_input[68896];
	for(int i = 0; i < 68896; i++)
		tmp_input[i] = i;

	// input token
	for(int i = 0; i < 68896; i++)
		packed_data[i + 105] = tmp_input[i];

	generateQKVMeta(packed_data, Q_ADDR_OFFSET, Q_META_OFFSET, seq_len, in_dim);
	generateQKVMeta(packed_data, K_ADDR_OFFSET, K_META_OFFSET, seq_len, in_dim);
	generateQKVMeta(packed_data, V_ADDR_OFFSET, V_META_OFFSET, seq_len, in_dim);

	generateMulQKMeta(packed_data, QK_ADDR_OFFSET, QK_META_OFFSET, seq_len, in_dim);

	generateSoftmaxMeta(packed_data, SOFTMAX_ADDR_OFFSET, SOFTMAX_META_OFFSET, seq_len, in_dim);

	generateMulSVMeta(packed_data, SV_ADDR_OFFSET, SV_META_OFFSET, seq_len, in_dim);

	printf("attention input data created !@!@!@!@!@!@!@\n");

	// memAccess(void *data, uint32_t addr, bool is_write, uint32_t transfer_size)
	// weight pre-load
	memAccess(W, Q_ADDR_OFFSET, 1, sizeof(uint16_t) * weight_size * 3);
	// memAccess(Wk, K_ADDR_OFFSET, 1, sizeof(uint16_t) * weight_size);
	// memAccess(Wv, V_ADDR_OFFSET, 1, sizeof(uint16_t) * weight_size);
}

bool pimblasMulPreprocess(PIMKernel *micro_kernel, int len, uint8_t **in0, uint8_t **in1) {
	return true;
}

bool pimblasReluPreprocess(PIMKernel *micro_kernel, int len, uint8_t **in) {
	return true;
}

bool pimblasBn1dPreprocess(PIMKernel *micro_kernel, int len_batch, int len_feature, uint8_t **w_mul,
						  uint8_t **w_add) {
	return true;
}

bool pimblasGemvPreprocess(struct PimCmdMetadata *cmd_metadata0, struct PimCmdMetadata *cmd_metadata1,
						   int len_in, int len_out, uint8_t **w) {
	int len_in_ = Ceiling(len_in, 8);
	int len_out_ = Ceiling(len_out, 4096);
	
	int w_idx = 0;   // mem
	int out_idx = 1; // mem
	int in_idx = 0;  // data

	int w_addr = 0;
	int out_addr = w_addr + len_in_ * len_out_ / UNITS_PER_WORD;
	int in_addr = 0;

	uint8_t tmp_input[5000];
	for(int i = 0; i < 5000;i++)
		tmp_input[i] = i;

	*w = GemvReshape(*w, len_in, len_out);
	*w = Transpose(*w, len_in, len_out);

	// memAccess(w, w_addr, true, (uint32_t)(len_in_ * len_out_ * UNIT_SIZE));
	
	cmd_metadata0->n_iter = (len_in_ + 8 - 1) / 8;
	cmd_metadata0->n_cmdgroup = 3;
	cmd_metadata0->operand[w_idx] = w_addr;
	cmd_metadata0->operand[out_idx] = out_addr;
	cmd_metadata0->data[in_idx] = in_addr;

	// Read SB → AB (Automatic in FPGA, ok)

	// Write ukernel code in Metadata (Only Send Data to FPGA)
	cmd_metadata0->code[0] = 0b10100100001000001000100000000000; // MAC(A)  GRF_B[A0]  BANK      SRF_M[A0]
	cmd_metadata0->code[1] = 0b00010000000001000000100000000111; // JUMP    -1         7
	cmd_metadata0->code[2] = 0b00100000000000000000000000000000; // EXIT

	// Write Input to SRF
	cmd_metadata0->cmd_info[0].op_code_tg = 0;  // RD
	cmd_metadata0->cmd_info[0].addr_tg = IDX_SRF;
	cmd_metadata0->cmd_info[0].data_tg = in_idx;
	cmd_metadata0->cmd_info[0].addr_tg_step = 0;
	cmd_metadata0->cmd_info[0].data_tg_step = 16;  // 16 Byte
	cmd_metadata0->cmd_info[0].n_cmd = 8;
	
	cmd_metadata0->input = tmp_input;

	printf("input data created !@!@!@!@!@!@!@\n");
	
	// Change to AB-PIM Mode
	cmd_metadata0->cmd_info[1].op_code_tg = 0;  // RD
	cmd_metadata0->cmd_info[1].addr_tg = IDX_PIM_OP_MODE;
	cmd_metadata0->cmd_info[1].data_tg = 0;
	cmd_metadata0->cmd_info[1].addr_tg_step = 0;
	cmd_metadata0->cmd_info[1].data_tg_step = 0;
	cmd_metadata0->cmd_info[1].n_cmd = 1;

	// Read Weight 8 Col
	cmd_metadata0->cmd_info[2].op_code_tg = 0;  // RD
	cmd_metadata0->cmd_info[2].addr_tg = w_idx + 16;
	cmd_metadata0->cmd_info[2].data_tg = 0;
	cmd_metadata0->cmd_info[2].addr_tg_step = 1 << co_pos;
	cmd_metadata0->cmd_info[2].data_tg_step = 0;
	cmd_metadata0->cmd_info[2].n_cmd = 8;
	
	cmd_metadata1->n_iter = 1;
	cmd_metadata1->n_cmdgroup = 2;
	cmd_metadata1->operand[w_idx] = w_addr;
	cmd_metadata1->operand[out_idx] = out_addr;
	cmd_metadata1->data[in_idx] = in_addr;
	cmd_metadata1->input_size = 4000; // input tmp size
	
	cmd_metadata1->input = tmp_input; // tmp input data
		
	cmd_metadata1->code[0] = 0b01000000100000000000000000000000; // MOV     BANK       GRF_B[0]
	cmd_metadata1->code[1] = 0b00100000000000000000000000000001; // EXIT
	// Read SB → AB (Automatic in FPGA, ok)
	
	// Change to AB-PIM Mode
	cmd_metadata1->cmd_info[0].op_code_tg = 0;  // RD
	cmd_metadata1->cmd_info[0].addr_tg = IDX_PIM_OP_MODE;
	cmd_metadata1->cmd_info[0].data_tg = 0;
	cmd_metadata1->cmd_info[0].addr_tg_step = 0;
	cmd_metadata1->cmd_info[0].data_tg_step = 0;
	cmd_metadata1->cmd_info[0].n_cmd = 1;
	
	// Write Output 1 Col
	cmd_metadata1->cmd_info[1].op_code_tg = 1;  // WR
	cmd_metadata1->cmd_info[1].addr_tg = out_idx;
	cmd_metadata1->cmd_info[1].data_tg = 0;
	cmd_metadata1->cmd_info[1].addr_tg_step = 1 << co_pos;
	cmd_metadata1->cmd_info[1].data_tg_step = 0;
	cmd_metadata1->cmd_info[1].n_cmd = 1;

	return true;
}

bool pimblasLstmPreprocess(PIMKernel *micro_kernel, int len_in, int len_out, uint8_t **w, uint8_t **b) {
	return true;
}

bool pim_add(struct PimCmdMetadata cmd_metadata, int len, uint8_t *in0, uint8_t *in1, uint8_t *out) {
	CmdGenExecute(cmd_metadata);
	return true;
}

bool pim_mul(PIMKernel micro_kernel, int len, uint8_t *in0, uint8_t *in1, uint8_t *out) {
	return true;
}

bool pim_relu(PIMKernel micro_kernel, int len, uint8_t *in0, uint8_t *out) {
	return true;
}

bool pim_bn1d(PIMKernel micro_kernel, int len_batch, int len_feature, uint8_t *in, uint8_t *w_mul,
			  uint8_t *w_add, uint8_t *out) {
	return true;
}

bool pim_FFN(uint32_t *packed_data, int seq_len, int in_dim, uint16_t *output){
    pimExecution(0, packed_data, 1);

	// read data
	memAccess(output, 0x00000000C0000000, 0, 100);

	printf("output : %u %u %u %u %u\n", output[0], output[1], output[2], output[3], output[4]);

    return true;
}

// void *data, uint32_t addr, bool is_write, uint32_t transfer_size

bool pim_attention(uint32_t *packed_data, int seq_len, int in_dim, uint16_t *output){

	printf("pimExecution start\n");

    pimExecution(0, packed_data, 1);

	printf("memory read start\n");

	// read data
	memAccess(output, 0x00000000C0000000, 0, 2 * 768 * 128);

	// printf("output 2: %u \n", output.size());

    return true;
}

bool pim_gemv(struct PimCmdMetadata cmd_metadata0, struct PimCmdMetadata cmd_metadata1, int len_in, int len_out, uint8_t *in, uint8_t *weight, uint8_t *out) {
	cmd_metadata0.input = (uint8_t *)calloc(len_in + 16, UNIT_SIZE);
	for (int i=0; i< len_in * UNIT_SIZE; i++) {
		cmd_metadata0.input[i + 16] = in[i];  // +16 for SRF_A (We use SRF_M!)	
	}
	cmd_metadata0.input_size = len_in * UNIT_SIZE;
	cmd_metadata1.input_size = 0;

	int code_iter = 2 * ((len_out + 4096 - 1) / 4096);
	for (int i=0; i< code_iter; i++) {
		CmdGenExecute(cmd_metadata0);
		CmdGenExecute(cmd_metadata1);
	}

	return true;
}

bool pim_lstm(PIMKernel micro_kernel, int len_in, int len_out, uint8_t *in, uint8_t *weight, uint8_t *bias, 
			  uint8_t *out) {
	return true;
}

uint8_t *Bn1dReshape(uint8_t *w, int l, int f) {
	uint8_t *w_ = (uint8_t*)calloc(l * f, UNIT_SIZE);

	for (int fi=0; fi<f; fi++)
		for (int li=0; li<l; li++)
			((uint16_t*)w_)[fi + li*f] = ((uint16_t*)w)[fi];

	return w_;	
}

uint8_t *GemvReshape(uint8_t *w, int m, int n) {
	int m_ = Ceiling(m, 8);
	int n_ = Ceiling(n, 4096);
	uint8_t *w_ = (uint8_t *)malloc(sizeof(uint16_t) * m_ * n_);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			((uint16_t *)w_)[i * m_ + j] = ((uint16_t *)w)[i * m + j];
		}
	}
	return w_;
}

uint8_t *LstmReshape(uint8_t *w, int m, int n) {
	int m_ = Ceiling(m, 8);
	int n_ = Ceiling(n, 4096);
	uint8_t *w_ = (uint8_t *)malloc(sizeof(uint16_t) * m_ * n_);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			((uint16_t *)w_)[i * m_ + j] = ((uint16_t *)w)[i * m + j];
		}
	}
	return w_;
}

uint8_t *Transpose(uint8_t *w, int m, int n) {
	m = Ceiling(m, 8);
	n = Ceiling(n, 4096);
	uint8_t *w_ = (uint8_t *)malloc(sizeof(uint16_t) * m * n);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			((uint16_t *)w_)[j * n + i] = ((uint16_t *)w)[i * m + j];
		}
	}
	return w_;
}
