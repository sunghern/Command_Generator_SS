#include <fcntl.h>
#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <stdbool.h>

//#include "PIM_SS_CG/fpga_runtime.h"
// #include "PIM_SoftwareStack_backup/fpga_runtime.h"
#include "../../xdma/cdev_sgdma.h"
#include "fpga_runtime.h"

// >> KKM
// #include "PIM_SoftwareStack/fpga_pim.h"
// KKM << 

// #include "dma_utils.c"

// ==============================================================
// Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC v2019.1 (64-bit)
// Copyright 1986-2019 Xilinx, Inc. All Rights Reserved.
// ==============================================================
// ctrl
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read/COR)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read)
//        bit 7  - auto_restart (Read/Write)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0  - Channel 0 (ap_done)
//        bit 1  - Channel 1 (ap_ready)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/TOW)
//        bit 0  - Channel 0 (ap_done)
//        bit 1  - Channel 1 (ap_ready)
//        others - reserved
// 0x10 : Data signal of ap_return
//        bit 31~0 - ap_return[31:0] (Read)
// 0x18 : Data signal of local_addr_scalar
//        bit 31~0 - local_addr_scalar[31:0] (Read/Write)
// 0x1c : reserved
// 0x20 : Data signal of local_addr
//        bit 31~0 - local_addr[31:0] (Read/Write)
// 0x24 : reserved
// 0x60 : Data signal of iswrite
//        bit 0  - iswrite[0] (Read/Write)
//        others - reserved
// 0x64 : reserved
// 0x40 ~
// 0x5f : Memory 'data' (32 * 8b)
//        Word n : bit [ 7: 0] - data[4n]
//                 bit [15: 8] - data[4n+1]
//                 bit [23:16] - data[4n+2]
//                 bit [31:24] - data[4n+3]
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XPIMCONTROLLER_CTRL_ADDR_AP_CTRL                0x00
#define XPIMCONTROLLER_CTRL_ADDR_GIE                    0x04
#define XPIMCONTROLLER_CTRL_ADDR_IER                    0x08
#define XPIMCONTROLLER_CTRL_ADDR_ISR                    0x0c
#define XPIMCONTROLLER_CTRL_ADDR_AP_RETURN              0x10
#define XPIMCONTROLLER_CTRL_BITS_AP_RETURN              32
#define XPIMCONTROLLER_CTRL_ADDR_LOCAL_ADDR_SCALAR_DATA 0x18
#define XPIMCONTROLLER_CTRL_BITS_LOCAL_ADDR_SCALAR_DATA 32
#define XPIMCONTROLLER_CTRL_ADDR_LOCAL_ADDR_DATA        0x20
#define XPIMCONTROLLER_CTRL_BITS_LOCAL_ADDR_DATA        32
#define XPIMCONTROLLER_CTRL_ADDR_ISWRITE_DATA           0x60
#define XPIMCONTROLLER_CTRL_BITS_ISWRITE_DATA           1
#define XPIMCONTROLLER_CTRL_ADDR_DATA_BASE              0x40
#define XPIMCONTROLLER_CTRL_ADDR_DATA_HIGH              0x5f
#define XPIMCONTROLLER_CTRL_WIDTH_DATA                  8
#define XPIMCONTROLLER_CTRL_DEPTH_DATA                  32
#define XPIMCONTROLLER_CTRL_ADDR_WRITE_RESULT_DATA      0x68
#define XPIMCONTROLLER_CTRL_BITS_WRITE_RESULT_DATA      1

#define WRITE_DEVICE_NAME_DEFAULT "/dev/xdma0_h2c_0"
#define READ_DEVICE_NAME_DEFAULT "/dev/xdma0_c2h_0"
#define WRITE_DEVICE_NAME_SECOND "/dev/xdma0_h2c_1"
#define READ_DEVICE_NAME_SECOND "/dev/xdma0_c2h_1"

#define AXI_LITE_INTERFACE_DEVICE "/dev/xdma0_user"
#define SIZE_DEFAULT (32)
#define COUNT_DEFAULT (1)

#define SBMR_LOCAL_ADDR         0x3fff
#define ABMR_LOCAL_ADDR         0x3ffe
#define PIM_OP_MODE_LOCAwrite_fpga_fd_gE    0
#define SINGLE_BANK_MODE    0
#define ALL_BANK_MODE       1
#define ALL_BANK_PIM_MODE   2

#define NO_IS_WRITE_RESULT_DATA  0
#define IS_WRITE_RESULT_DATA     1

#define WRITE 1

#define IS_SCALAR                (1)
#define IS_VECTOR                (0)
#define RETURN_OFFSET            (0x44A00000 + 0x10)
#define LOCAL_ADDR_OFFSET        (0x44A00000 + 0x18)
#define WRITE_GRF_OFFSET         (0x44A00000 + 0x20)
#define XPIMCONTROLLER_CTRL_ADDR_PACKED_DATA_V_BASE (0x44A00000 + 0x8000)
#define ENABLE_OFFSET            (0x44A00000 + 0)
#define CUSTOM_IP_OFFSET         (0x44A00000)
#define ADDR_OFFSET              (0x10)
#define DATA_OFFSET              (0x18)
#define IS_WRITE                 (1)
#define ENABLE_DATA              (1)
#define IS_BOOL                  (1)
#define NOT_BOOL                 (0)
#define INT_SIZE                 (32)
#define BOOL_SIZE                (1)
#define CHAR_SIZE                (8)
#define NORMAL_WRITE             (0)
#define DONE_DATA                (0b00000010)
#define COLUMN_SHIFT             (8)

#define BANK_SOURCE_ADDR_1 (unsigned int)0b00000000000000000010000000000000
#define BANK_SOURCE_ADDR_2 (unsigned int)0b00000000000000000010100100000000
#define BANK_SOURCE_ADDR_3 (unsigned int)0b00000000000000000011001000000000
#define BANK_SOURCE_ADDR_4 (unsigned int)0b00000000000000000011101100000000
#define BANK_SOURCE_ADDR_5 (unsigned int)0b00000000000000000000000000010000

#define RESULT_BRAM_ADDR    (0xC0000000)
#define CRF_BRAM_ADDR       (0xC2000000)
#define GRF_A_BRAM_ADDR     (0xC4000000)
#define GRF_B_BRAM_ADDR     (0xC6000000)
#define SRF_A_BRAM_ADDR     (0xC8000000)
#define SRF_M_BRAM_ADDR     (0xCA000000)
#define BANK_ADDR_BRAM_ADDR (0xCC000000)
#define MAP_OPEN_LENGTH     (0xF0000000)

// register local_address
#define SBMR_LOCAL_ADDR             0x3fff
#define ABMR_LOCAL_ADDR             0x3ffe
#define PIM_OP_MODE_LOCAL_ADDR      0x3ffd
#define CRF_LOCAL_ADDR              0x3ffc
#define GRF_A_B_LOCAL_ADDR          0x3ffb
#define SRF_A_M_LOCAL_ADDR          0x3ffa
#define BANK_ADDR_REG_ADDR          0x3ff9
#define PACKED_DATA_TRANSFER_ADDR   0x3ff8

union singleToQuad{
    struct{
        uint32_t first      : 8;
        uint32_t second     : 8;
        uint32_t third      : 8;
        uint32_t forth      : 8;
    };
    unsigned int combine_byte;
};

union DoubleToQuad{
    struct{
        uint32_t first      : 16;
        uint32_t second     : 16;
    };
    unsigned int combine_byte;
};

union bit_split{
    struct{
        unsigned int ch : 4;
        unsigned int ba : 2;
        unsigned int bg : 2;
        unsigned int col : 5;
        unsigned int row : 19;
    };
    unsigned int address;
};

// struct CmdInfo{
//     uint8_t op_code_tg;
//     uint8_t addr_tg;
//     uint8_t data_tg;
//     uint16_t addr_tg_step;
//     uint16_t data_tg_step;
//     uint16_t n_cmd;
// };

// struct PimCgInitData{
//     uint32_t pim_reg[16];
//     uint8_t pim_op[16];
// };

// struct PimCgMetaData{
//     uint16_t n_iter;
//     uint16_t n_cmdgroup;
//     uint32_t operand[16];
//     uint32_t data[16];
//     uint32_t code[32];
//     uint8_t *input;
//     uint32_t input_size;
//     uint32_t init_data[4096];
//     struct CmdInfo cmd_info[16];
// };

static struct option const long_opts[] = {
    {"device", required_argument, NULL, 'd'},
    {"address", required_argument, NULL, 'a'},
    {"aperture", required_argument, NULL, 'k'},
    {"size", required_argument, NULL, 's'},
    {"offset", required_argument, NULL, 'o'},
    {"count", required_argument, NULL, 'c'},
    {"data infile", required_argument, NULL, 'f'},
    {"data outfile", required_argument, NULL, 'w'},
    {"help", no_argument, NULL, 'h'},
    {"verbose", no_argument, NULL, 'v'},
    {0, 0, 0, 0}
};

static int start_check_g = 0;
static int bank_mode_g = 0;
static int eop_flush_g = 0;

static int write_fd_flag_g = 0;
static int read_fd_flag_g = 0;

static int write_fpga_fd_g = 0;
static int read_fpga_fd_g = 0;

static int write_fpga_fd_g_1 = 0;
static int read_fpga_fd_g_1 = 0;

int verbose = 0;
static int check_g;
static bool grf_write_flag_g = 0;

void timespec_sub(struct timespec *t1, struct timespec *t2);
static int timespec_check(struct timespec *t);

unsigned int readData(char *fname, int fd, off_t offset, uint32_t *buf, uint8_t *scalar_buf, uint32_t size, uint8_t is_enable);
void writeData(char *fname, int fd, off_t write_addr, uint32_t *vector_data, unsigned int scalar_data, bool is_scalar, bool is_enable, int size);

static uint64_t dataToFpga(uint32_t addr, uint32_t *data, uint64_t aperture, uint64_t size);

static int dataFromFpga(uint32_t addr, uint32_t *data, uint64_t aperture, uint64_t size, uint64_t offset);

uint64_t pimExecution(uint32_t addr, uint32_t *data, int iswrite);
void memAccess(uint16_t *data, uint32_t addr, bool is_write, uint32_t transfer_size);

bool CmdGeninit(struct PimCgInitData cg_init_data);
bool CmdGenExecute(struct PimCmdMetadata cmd_metadata);

bool CmdGenInit(struct PimCgInitData cg_init_data){

    // init_data[0] ~ init_data[19] : init data
    // init_data[20] ~ init_data[98] : reserved
    // init_data[99] : is_meta_data // 0 : init_data, 1 : meta_data
    uint32_t init_data[5000];

    // command generator initiation
    init_data[99] = 0;

    printf("command generate init\n");

    // copy reg data
    for(int i = 0; i < 16; i++)
        init_data[i] = cg_init_data.pim_reg[i];

    // copy op data
    for(int j = 0; j < 4; j++){
        for(int i = 0; i < 4; i++){
            init_data[16 + j] = cg_init_data.pim_op[j*4+i] | 
                            (cg_init_data.pim_op[j*4+i+1] << 8) | 
                            (cg_init_data.pim_op[j*4+i+2] << 16) | 
                            (cg_init_data.pim_op[j*4+i+3] << 24);
        }
    }

    // data write to FPGA
    // pimExecution(0, init_data, 1); // addr need to be modified

    return 1;
}

bool CmdGenExecute(struct PimCmdMetadata cg_meta_data){

    struct timespec ts_start, ts_end, runtime_start, runtime_end;
    int rc;

    printf("cmdGenExecution start\n");

    rc = clock_gettime(CLOCK_MONOTONIC, &ts_start);
    // meta_data[0] ~ meta_data[52] : meta data
    // meta_data[53] ~ meta_data[98] : reserved
    // meat_data[99] : is_meta_data // 0 : init_data, 1 : meta_data
    uint32_t meta_data[150000]; // to be modified

    for(int i = 0; i < 150000; i++)
        meta_data[i] = 10;

    // copy iter and cmd group
    meta_data[0] = cg_meta_data.n_iter | 
                    (cg_meta_data.n_cmdgroup << 8);

    // copy operand addr
    for(int i = 0; i < 16; i++){
        meta_data[i+1] = cg_meta_data.operand[i];
        meta_data[i+17] = cg_meta_data.data[i];
    }

    // copy addr tg info 
    for(int j = 0; j < 4; j++){
        for(int i = 0; i < 4; i++){
            meta_data[33 + j] = cg_meta_data.cmd_info[j*4+i].op_code_tg | 
                            (cg_meta_data.cmd_info[j*4+i+1].op_code_tg << 8) | 
                            (cg_meta_data.cmd_info[j*4+i+2].op_code_tg << 16) | 
                            (cg_meta_data.cmd_info[j*4+i+3].op_code_tg << 24);  

            meta_data[37 + j] = cg_meta_data.cmd_info[j*4+i].addr_tg | 
                            (cg_meta_data.cmd_info[j*4+i+1].addr_tg << 8) | 
                            (cg_meta_data.cmd_info[j*4+i+2].addr_tg << 16) | 
                            (cg_meta_data.cmd_info[j*4+i+3].addr_tg << 24);

            meta_data[41 + j] = cg_meta_data.cmd_info[j*4+i].data_tg | 
                            (cg_meta_data.cmd_info[j*4+i+1].data_tg << 8) | 
                            (cg_meta_data.cmd_info[j*4+i+2].data_tg << 16) | 
                            (cg_meta_data.cmd_info[j*4+i+3].data_tg << 24);
        }
    }

    // printf("command generate metadata 4\n");

    // copy data tg step info 
    for(int j = 0; j < 8; j++){
        for(int i = 0; i < 2; i++){
            meta_data[45 + j] = cg_meta_data.cmd_info[j*2+i].addr_tg_step | 
                            (cg_meta_data.cmd_info[j*2+i+1].addr_tg_step << 16);

            meta_data[53 + j] = cg_meta_data.cmd_info[j*2+i].data_tg_step | 
                            (cg_meta_data.cmd_info[j*2+i+1].data_tg_step << 16);

            meta_data[61 + j] = cg_meta_data.cmd_info[j*2+i].n_cmd | 
                            (cg_meta_data.cmd_info[j*2+i+1].n_cmd << 16);
        }
    }

    for(int i = 0; i < 32; i++)
        meta_data[69+i] = cg_meta_data.code[i];

     // command generator trigger
    meta_data[101] = 1;

    for(int i = 0; i < (cg_meta_data.input_size/4); i++){
        meta_data[102+i] = cg_meta_data.input[i*4] | 
                            cg_meta_data.input[i*4+1] << 8 | 
                            cg_meta_data.input[i*4+2] << 16 | 
                            cg_meta_data.input[i*4+3] << 24;  
        
        cg_meta_data.input[i];
    }
    // return 0;
    // for(int i = 0; i < 100; i++)
    //     printf("input data : %u\n", meta_data[i]);

    for(int i = 69; i < 100; i++)
        meta_data[i] = 0b10100100001000001000100000000000;

	meta_data[0] = 327685;
	meta_data[61] = 327685;
	meta_data[62] = 327685;
	meta_data[63] = 327685;
	meta_data[64] = 327685;
	meta_data[65] = 327685;
	meta_data[66] = 327685;
	meta_data[67] = 327685;
	meta_data[68] = 327685;
    meta_data[102] = 1;

    uint32_t crf_data[32];

    crf_data[0] = 0b10100100000000000000000000101010;
    crf_data[1] = 0b10100100000000000000000000101010;
    crf_data[2] = 0b10100100000000000000000000101010;
	crf_data[3] = 0b10100100000000000000000000101010;
    crf_data[4] = 0b00100100000000000000000000000010;

    for(int i = 0; i < 32; i++)
        meta_data[i+69] = crf_data[i];


    //printf("input size : %u\n", cg_meta_data.input_size);

    pimExecution(0, meta_data, 1); // addr need to be modified

    //printf("pimexecution end\n");

    rc = clock_gettime(CLOCK_MONOTONIC, &ts_end);

    uint64_t execution_time = (ts_end.tv_sec - ts_start.tv_sec) + (ts_end.tv_nsec - ts_start.tv_nsec);
    printf("**************************** FPGA runtime latency ***************************: %llu\n", execution_time);

    return 1;
}

void memAccess(uint16_t *data, uint32_t addr, bool is_write, uint32_t transfer_size){
    
    write_fpga_fd_g_1 = open(WRITE_DEVICE_NAME_SECOND, O_RDWR);   
    read_fpga_fd_g_1 = open(READ_DEVICE_NAME_SECOND, O_RDWR);

    uint8_t done_check;
    // uint16_t *tmp;

    uint16_t *tmp_buf;
    tmp_buf = (uint16_t *)malloc(sizeof(uint16_t) * transfer_size);

    // for(int i = 0; i < transfer_size; i++)
    //     *(tmp_buf+i) = rand();

    if(is_write == 1){
        writeData(WRITE_DEVICE_NAME_SECOND, write_fpga_fd_g_1, addr, data, ENABLE_DATA, IS_VECTOR, 0, transfer_size);
    }else{
        // readData(READ_DEVICE_NAME_SECOND, read_fpga_fd_g_1, addr, tmp, &done_check, transfer_size, IS_VECTOR);
        for(int i = 0; i < transfer_size; i++)
            *(tmp_buf+i) = rand();
    }

    // free(tmp_buf);
}
// char *fname, int fd, off_t offset, uint32_t *buf, uint8_t *scalar_buf, uint32_t size, uint8_t is_scalar)
// char *fname, int fd, off_t write_addr, uint32_t *vector_data, unsigned int scalar_data, bool is_scalar, bool is_enable, int size

// union DoubleToQuad{
//  struct{
//      uint32_t first      : 16;
//      uint32_t second     : 16;
//  };
//  unsigned int combine_byte;
// };

uint64_t pimExecution(uint32_t addr, uint32_t *data, int iswrite)
{
    struct timespec ts_start, ts_end;
    union bit_split bs;
    int infile_fd = -1;
    int outfile_fd = -1; // open device(c2h, h2c channel)
    long total_time = 0;
    float result; 
    uint64_t aperture = 0;

    write_fpga_fd_g = open(WRITE_DEVICE_NAME_DEFAULT, O_RDWR);   
    read_fpga_fd_g = open(READ_DEVICE_NAME_DEFAULT, O_RDWR);

    //printf("pim execution!@#!@$!@$!@$!@$!@$!@4\n");

    if(iswrite == WRITE){
        return dataToFpga(addr, data, aperture, 5000);
    }else{
        return dataFromFpga(addr, data, aperture, 5000, 0);
    }
}

static uint64_t dataToFpga(uint32_t addr, uint32_t *data, uint64_t aperture, uint64_t size){

    size_t bytes_done = 0;
    uint32_t *buffer = data;
    char *allocated = NULL;
    struct timespec ts_start, ts_end, runtime_start, runtime_end;
    union bit_split bs;
    int infile_fd = -1;
    int outfile_fd = -1; // open device(c2h, h2c channel)
    long total_time = 0;
    float result;
    int underflow = 0;
    uint32_t polling_count = 0;
    int rc;

    // unsigned int row_local_addr = 0;
    // unsigned int col_local_addr = 0;

    // bs.address = addr;

    // row_local_addr = bs.row;
    // col_local_addr = bs.col;
 
    // aperture : ?
    if (aperture) {
        struct xdma_aperture_ioctl io;

        io.buffer = (unsigned long)buffer;
        io.len = size;
        io.done = 0UL;

        rc = ioctl(write_fpga_fd_g, IOCTL_XDMA_APERTURE_W, &io);
        if (rc < 0 || io.error) {
            fprintf(stdout,
            "#%d: aperture W ioctl failed %d,%d.\n",
            io, rc, io.error);
            goto out;
        }
        bytes_done = io.done;
    }else{ // else : write data to fpga

        uint8_t done_check = 0;

        // row addr - have to be changed ***************************************************
        // row_local_addr = PACKED_DATA_TRANSFER_ADDR;

        // if(row_local_addr == PACKED_DATA_TRANSFER_ADDR){ // data transfer to PIM Module

        // printf("opcode - pimExecution : %u\n", data[0]);

        // for(int i = 0; i < 200; i++){
        //     printf("%u ", data[i+1]);
        // }

        // printf("\n");

        // change to mul
        // data[2+9] = 2450030592;

        // if(data[0] == 1){
        //     uint32_t tmp_buf[40];
        //     for(int i = 0; i < 40; i++)
        //         tmp_buf[i] = data[41+i];

        //     for(int j = 0; j < 36; j++)
        //         for(int i = 0; i < 40; i++)
        //             data[201 + i + 40*j] = tmp_buf[i];
        // }

        // lstm - data[4999] = 1 --> count:w 
        // lstm opcode == 4, 14
        // 0b0000000000000000000000000001 0100
        // buffer[0] = 0b00000000000000000000000000010100;
        // buffer[0] = 0b00000000000000000000000000010010;

        // count clock
        // if((buffer[0] == 0b00000000000000000000000000010100)||(buffer[0] == 0b00000000000000000000000000010010)){
        //     buffer[4999] = 1;
        //     printf("buffer[4999] : %u\n", buffer[4999]);
        // }else{ // not count clock
        //     buffer[4999] = 0;
        //     printf("buffer[4999] : %u\n", buffer[4999]);
        // } 

        // FILE *fp;
        // fp = fopen("command_gen.txt", "w+");

        // if(fp == NULL){
        //     printf("file open error\n");
        // }else{
        //     printf("file open success\n");
        // }

        // for(int i = 0; i < 5000; i++)
        //     fprintf(fp, "%x\n", buffer[i]);

        // fclose(fp);  0x0000000100000000

        //writeData(char *fname, int fd, off_t write_addr, uint32_t *vector_data, unsigned int scalar_data, bool is_scalar, bool is_enable, int size)

        // ch 0   
        writeData(WRITE_DEVICE_NAME_DEFAULT, write_fpga_fd_g, 0x0018 + 0x0000000100000000, buffer, 0, IS_SCALAR, 0, 4);
        // ch 1
        writeData(WRITE_DEVICE_NAME_DEFAULT, write_fpga_fd_g, 0x0020 + 0x0000000100000000, buffer, 0x10000000, IS_SCALAR, 0, 4);
        // ch 2
        writeData(WRITE_DEVICE_NAME_DEFAULT, write_fpga_fd_g, 0x0028 + 0x0000000100000000, buffer, 0x20000000, IS_SCALAR, 0, 4);
        // ch 3
        writeData(WRITE_DEVICE_NAME_DEFAULT, write_fpga_fd_g, 0x0030 + 0x0000000100000000, buffer, 0x30000000, IS_SCALAR, 0, 4);
        // ch 4
        writeData(WRITE_DEVICE_NAME_DEFAULT, write_fpga_fd_g, 0x0038 + 0x0000000100000000, buffer, 0x40000000, IS_SCALAR, 0, 4);
        // ch 5
        writeData(WRITE_DEVICE_NAME_DEFAULT, write_fpga_fd_g, 0x0040 + 0x0000000100000000, buffer, 0x50000000, IS_SCALAR, 0, 4);
        // ch 6
        writeData(WRITE_DEVICE_NAME_DEFAULT, write_fpga_fd_g, 0x0048 + 0x0000000100000000, buffer, 0x60000000, IS_SCALAR, 0, 4);
        // ch 7
        writeData(WRITE_DEVICE_NAME_DEFAULT, write_fpga_fd_g, 0x0050 + 0x0000000100000000, buffer, 0x70000000, IS_SCALAR, 0, 4);
        // ch 8
        writeData(WRITE_DEVICE_NAME_DEFAULT, write_fpga_fd_g, 0x0058 + 0x0000000100000000, buffer, 0x80000000, IS_SCALAR, 0, 4);
        // ch 9
        writeData(WRITE_DEVICE_NAME_DEFAULT, write_fpga_fd_g, 0x0060 + 0x0000000100000000, buffer, 0x90000000, IS_SCALAR, 0, 4);
        // ch 10
        writeData(WRITE_DEVICE_NAME_DEFAULT, write_fpga_fd_g, 0x0068 + 0x0000000100000000, buffer, 0xA0000000, IS_SCALAR, 0, 4);
        // ch 11
        writeData(WRITE_DEVICE_NAME_DEFAULT, write_fpga_fd_g, 0x0070 + 0x0000000100000000, buffer, 0xB0000000, IS_SCALAR, 0, 4);
        // ch 12
        writeData(WRITE_DEVICE_NAME_DEFAULT, write_fpga_fd_g, 0x0078 + 0x0000000100000000, buffer, 0xC0000000, IS_SCALAR, 0, 4);
        // ch 13
        writeData(WRITE_DEVICE_NAME_DEFAULT, write_fpga_fd_g, 0x0080 + 0x0000000100000000, buffer, 0xD0000000, IS_SCALAR, 0, 4);
        // ch 14
        writeData(WRITE_DEVICE_NAME_DEFAULT, write_fpga_fd_g, 0x0088 + 0x0000000100000000, buffer, 0xE0000000, IS_SCALAR, 0, 4);

        // packed data setting 
        writeData(WRITE_DEVICE_NAME_DEFAULT, write_fpga_fd_g, 0x80000 + 0x0000000100000000, buffer, ENABLE_DATA, IS_VECTOR, 0, 100000*sizeof(uint32_t));
        // data setting enable
        writeData(WRITE_DEVICE_NAME_DEFAULT, write_fpga_fd_g, 0x0000000100000000, buffer, ENABLE_DATA, IS_SCALAR, 1, sizeof(uint8_t));

        // printf("trigger data : %d\n", buffer[99]);
        // get start time
        rc = clock_gettime(CLOCK_MONOTONIC, &ts_start);

        while(!(done_check & 0b00000010)){
            readData(READ_DEVICE_NAME_DEFAULT, read_fpga_fd_g, 0x0000000100000000, buffer, &done_check, sizeof(uint8_t), IS_SCALAR);
            // printf("done_check : %d\n", done_check);
            polling_count = polling_count + 1;
        }

        rc = clock_gettime(CLOCK_MONOTONIC, &ts_end);

        printf("polling check : %u\n", polling_count);
        // printf("time : %");

        uint32_t tmp_ret;

        readData(READ_DEVICE_NAME_DEFAULT, read_fpga_fd_g, 0x10 + 0x100000000, &tmp_ret, 0, sizeof(uint32_t), IS_VECTOR);
        printf("return value : %u\n", tmp_ret);
        // get end time

        uint64_t execution_time = (ts_end.tv_sec - ts_start.tv_sec) + (ts_end.tv_nsec - ts_start.tv_nsec);
        printf("*******************************************command generator execution time ***********************************************: %llu\n", execution_time);

        // printf("*********************************88\n");

        // }else{ // data transfer to normal bank
        //     writeData(WRITE_DEVICE_NAME_DEFAULT, write_fpga_fd_g, addr, buffer, 0, IS_VECTOR, 0, 8*sizeof(uint32_t));
        // }
    }

    if (rc < 0)
        goto out;

    bytes_done = rc;


    /* subtract the start time from the end time */
    // timespec_sub(&ts_end, &ts_start);
    // total_time += ts_end.tv_nsec;
    // /* a bit less accurate but side-effects are accounted for */
    // if (verbose)
    //     fprintf(stdout,
    //     "#%lu: CLOCK_MONOTONIC %ld.%09ld sec. write %ld bytes\n",
    //     i, ts_end.tv_sec, ts_end.tv_nsec, size);

    // uint64_t execution_time = (ts_end.tv_sec - ts_start.tv_sec) + (ts_end.tv_nsec - ts_start.tv_nsec);
    // printf("*******************************************execution time ***********************************************: %llu\n", execution_time);

    // printf("*******************************************op code ***********************************************: %u\n", data[0]);
    // printf("*******************************************polling_count ***********************************************: %u\n", polling_count);
    // printf("*******************************************execution time ***********************************************: %llu\n", execution_time);
    // if(row_local_addr == PACKED_DATA_TRANSFER_ADDR){
    //     // compute execution time
    //     uint64_t execution_time = (ts_end.tv_sec - ts_start.tv_sec) + (ts_end.tv_nsec - ts_start.tv_nsec);
    //     char tmp_str[50] = {0x00,};
    //     sprintf(tmp_str, "%llu", execution_time);

    //     FILE* fp = fopen("fpga_time.txt", "w+");

    //     if(fp == NULL){
    //         printf("file open fail\n");
    //     }else{
    //         printf("file open success\n");
    //     }
        
    //     fputs(tmp_str, fp);

    //     fclose(fp);

    //     return (execution_time);
    // }else{
    //     return 0;
    // }

out:
close(write_fpga_fd_g);
if (infile_fd >= 0)
close(infile_fd);
if (outfile_fd >= 0)
close(outfile_fd);
free(allocated);

if (rc < 0)
    return rc;
/* treat underflow as error */
return underflow ? -EIO : 0;
}

static int dataFromFpga(uint32_t addr, uint32_t *data, uint64_t aperture, uint64_t size, uint64_t offset)
{
    uint64_t i;
    ssize_t rc;
    size_t bytes_done = 0;
    size_t out_offset = 0;
    uint32_t *buffer = data;
    uint8_t scalar_buf;
    char *allocated = NULL;
    struct timespec ts_start, ts_end;
    int infile_fd = -1;
    int outfile_fd = -1;
    long total_time = 0;
    float result;
    float avg_time = 0;
    int underflow = 0;

    // device check
    if (read_fpga_fd_g < 0) {
        fprintf(stderr, "unable to open device %s, %d.\n",
        READ_DEVICE_NAME_DEFAULT, read_fpga_fd_g);
        perror("open device");
        return -EINVAL;
    }

    /* write buffer to AXI MM address using SGDMA */
    rc = clock_gettime(CLOCK_MONOTONIC, &ts_start);
    // aperture : ?
    if (aperture) {
        struct xdma_aperture_ioctl io;

        io.buffer = (unsigned long)buffer;
        io.len = size;
        io.ep_addr = addr;
        io.aperture = aperture;
        io.done = 0UL;

        rc = ioctl(read_fpga_fd_g, IOCTL_XDMA_APERTURE_W, &io);
        if (rc < 0 || io.error) {
            fprintf(stdout,
            "#%d: aperture W ioctl failed %d,%d.\n",
            i, rc, io.error);
            goto out;
        }

        bytes_done = io.done;
    } else { // else : write data to fpga
        readData(READ_DEVICE_NAME_DEFAULT, read_fpga_fd_g, addr, data, &scalar_buf, 8*sizeof(uint32_t), IS_VECTOR);

        if (rc < 0)
            goto out;

        bytes_done = rc;
    }

    rc = clock_gettime(CLOCK_MONOTONIC, &ts_end);

    long execution_time = (ts_end.tv_sec - ts_start.tv_sec) + (ts_end.tv_nsec - ts_start.tv_nsec);

    // printf("********** READ TIME ********** : %ld nanosecond\n", execution_time);

    if (verbose)
        fprintf(stdout,
        "#%lu: CLOCK_MONOTONIC %ld.%09ld sec. write %ld bytes\n",
        i, ts_end.tv_sec, ts_end.tv_nsec, size);

    if (!underflow) {
        avg_time = (float)total_time;
        result = ((float)size)*1000/avg_time;
    }

    return execution_time;

out:
close(read_fpga_fd_g);
if (infile_fd >= 0)
close(infile_fd);
if (outfile_fd >= 0)
close(outfile_fd);
free(allocated);

if (rc < 0)
    return rc;
/* treat underflow as error */
return underflow ? -EIO : 0;
}

void writeData(char *fname, int fd, off_t write_addr, uint32_t *vector_data, unsigned int scalar_data, bool is_scalar, bool is_enable, int size){

    ssize_t rc;
    uint8_t enable_buf = scalar_data;
    uint32_t local_addr_buf = scalar_data; 
    
    // writeData(WRITE_DEVICE_NAME_DEFAULT, write_fpga_fd_g, LOCAL_ADDR_OFFSET, buffer, 0, IS_SCALAR, 0, 4);
    if(is_scalar){
        if(is_enable){
            printf("enable\n");
            printf("enable write addr : %x\n", write_addr);
            rc = lseek(fd, write_addr, SEEK_SET);
            if (rc == -1) {
                fprintf(stderr, "%s, seek off 0x%lx != 0x%lx.\n",
                fname, rc, write_addr);
                perror("seek file");
                // return -EIO;
            }

            rc = write(fd, &enable_buf, sizeof(uint8_t));
            if (rc < 0) {
                printf("test write error\n");
                fprintf(stderr, "%s, write 0x%lx @ 0x%lx failed %ld.\n",
                fname, 4, write_addr, rc);
                perror("write file ");
                // return -EIO;
            }
        }else{
            printf("local addr lseek\n");
            printf("local write addr : %x\n", write_addr);
            rc = lseek(fd, write_addr, SEEK_SET);
            if (rc == -1) {
                fprintf(stderr, "%s, seek off 0x%lx != 0x%lx.\n",
                fname, rc, write_addr);
                perror("seek file");
                // return -EIO;
            }
            printf("local addr write\n");

            rc = write(fd, &local_addr_buf, sizeof(uint32_t));
            if (rc < 0) {
                printf("test write error\n");
                fprintf(stderr, "%s, write 0x%lx @ 0x%lx failed %ld.\n",
                fname, 4, write_addr, rc);
                perror("write file ");
                // return -EIO;
            }
            printf("local addr write finish\n");
        }
    }else{
        printf("data\n");
        printf("data write addr : %x\n", write_addr);

        rc = lseek(fd, write_addr, SEEK_SET);
        if (rc == -1) {
            fprintf(stderr, "%s, seek off 0x%lx != 0x%lx.\n",
            fname, rc, write_addr);
            perror("seek file");
            // return -EIO;
        }

        rc = write(fd, vector_data, size);
        if (rc < 0) {
            printf("test write error\n");
            fprintf(stderr, "%s, write 0x%lx @ 0x%lx failed %ld.\n",
            fname, 4, write_addr, rc);
            perror("write file ");
            // return -EIO;
        }
    }
}

unsigned int readData(char *fname, int fd, off_t offset, uint32_t *buf, uint8_t *scalar_buf, uint32_t size, uint8_t is_scalar)
{
    ssize_t rc;
    unsigned int data_buf;
    unsigned char enable_buf = 0;
    uint32_t *tmp_buf;
    uint32_t tmp;

    // temp fd, temp fname
    if(is_scalar == 0){

        printf("vector read start\n");
        printf("offset : %u\n", offset);

        rc = lseek(fd, offset, SEEK_SET);
        if (rc == -1) {
            fprintf(stderr, "%s, seek off 0x%lx != 0x%lx.\n",
            READ_DEVICE_NAME_DEFAULT, rc, offset);
            perror("seek file");
            // return -EIO;
        }  

        rc = read(fd, buf, size);
        if (rc < 0) {
            printf("test read error\n");
            fprintf(stderr, "%s, read 0x%lx @ 0x%lx failed %ld.\n",
            READ_DEVICE_NAME_DEFAULT, 4, offset, rc);
            perror("read file");
            // return -EIO;
        }
    }else{
        rc = lseek(fd, offset, SEEK_SET);
        if (rc == -1) {
            fprintf(stderr, "%s, seek off 0x%lx != 0x%lx.\n",
            READ_DEVICE_NAME_DEFAULT, rc, offset);
            perror("seek file");
            // return -EIO;
        }  

        rc = read(fd, scalar_buf, size);
        if (rc < 0) {
            printf("test read error\n");
            fprintf(stderr, "%s, read 0x%lx @ 0x%lx failed %ld.\n",
            READ_DEVICE_NAME_DEFAULT, 4, offset, rc);
            perror("read file");
            // return -EIO;
        }
     return data_buf;
    }
}

static int timespec_check(struct timespec *t)
{
    if ((t->tv_nsec < 0) || (t->tv_nsec >= 1000000000))
    return -1;
    return 0;

}

void timespec_sub(struct timespec *t1, struct timespec *t2)
{
    if (timespec_check(t1) < 0) {
    fprintf(stderr, "invalid time #1: %lld.%.9ld.\n",
    (long long)t1->tv_sec, t1->tv_nsec);
    return;
    }
    if (timespec_check(t2) < 0) {
    fprintf(stderr, "invalid time #2: %lld.%.9ld.\n",
    (long long)t2->tv_sec, t2->tv_nsec);
    return;
    }
    t1->tv_sec -= t2->tv_sec;
    t1->tv_nsec -= t2->tv_nsec;
    if (t1->tv_nsec >= 1000000000) {
    t1->tv_sec++;
    t1->tv_nsec -= 1000000000;
    } else if (t1->tv_nsec < 0) {
    t1->tv_sec--;
    t1->tv_nsec += 1000000000;
    }
}
