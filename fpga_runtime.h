#pragma once
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
#include "cmd_generator.h" // >> KKM for CG<<

// #include "../xdma/cdev_sgdma.h"
#ifdef __cplusplus
extern "C" {
#endif

void timespec_sub(struct timespec *t1, struct timespec *t2);
static int timespec_check(struct timespec *t);

unsigned int readData(char *fname, int fd, off_t offset, uint32_t *buf, uint8_t *scalar_buf, uint32_t size, uint8_t is_enable);

void writeData(char *fname, int fd, off_t write_addr, uint32_t *vector_data, unsigned int scalar_data, bool is_scalar, bool is_enable, int size);

static uint64_t dataToFpga(uint32_t addr, uint32_t *data, uint64_t aperture, uint64_t size);
static int dataFromFpga(uint32_t addr, uint32_t *data, uint64_t aperture, uint64_t size, uint64_t offset);

uint64_t pimExecution(uint32_t addr, uint32_t *data, int iswrite);
bool CmdGenInit(struct PimCgInitData cg_init_data);
bool CmdGenExecute(struct PimCmdMetadata cmd_metadata);
void memAccess(uint16_t *data, uint32_t addr, bool is_write, uint32_t transfer_size);

#ifdef __cplusplus
}
#endif
