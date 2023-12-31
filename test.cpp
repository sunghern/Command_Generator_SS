#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <chrono>
#include "pim_blas.h"
using half_float::half;
typedef unsigned short uint16;
typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;
typedef std::chrono::duration<float> fsec;

#define ABS(x) ((x < 0) ? (-x) : (x))

std::random_device random_device;
auto rng = std::mt19937(random_device());
auto f32rng = std::bind(std::normal_distribution<float>(0, 1), std::ref(rng));

void test_add_blas(int option) {
	if (DebugMode())
		std::cout << "LEN_PIM: " << LEN_PIM << std::endl;
	
	int n;// 1024 x 1024 x 32 → Tested OK!
	if (option == 1)
		n = 65536;
	else if (option == 2)
		n = 131072;
	else if (option == 3)
		n = 4194304;
	else
		std::cout << "choose option in [1, 2, 3]\n";

	uint8_t *in0 = (uint8_t *)malloc(sizeof(uint16_t) * n);
	uint8_t *in1 = (uint8_t *)malloc(sizeof(uint16_t) * n);
	uint8_t *out = (uint8_t *)malloc(sizeof(uint16_t) * n);
	uint8_t *ans = (uint8_t *)malloc(sizeof(uint16_t) * n);

	for (int i = 0; i < n; i++)	{
		((uint16_t *)in0)[i] = rand();
		((uint16_t *)in1)[i] = rand();
		((uint16_t *)ans)[i] = ((uint16_t *)in0)[i] + ((uint16_t *)in1)[i];
	}

	if (DebugMode())
		std::cout << "///// Preprocessing ADD BLAS... /////\n";

	struct PimCmdMetadata add_cmd_metadata;
	pimblasAddPreprocess(&add_cmd_metadata, n, &in0, &in1);

	if (DebugMode())
		std::cout << "///// Testing ADD BLAS... /////\n";
	
	auto start = Time::now();
	pim_add(add_cmd_metadata, n, in0, in1, out);
	auto end = Time::now();
	fsec time = end - start;
	std::cout << time.count() << "s\n";

	if (DebugMode())
		std::cout << "///// Test ADD BLAS Ended!! /////\n";

	int error = 0;
	for (int i = 0; i < n; i++)
		error = error + ABS(((uint16_t *)out)[i] - ((uint16_t *)ans)[i]);
	
	std::cout << "\n\nERROR: " << error << std::endl;

	return;
}

void test_mul_blas(int option) {
	if (DebugMode())
		std::cout << "LEN_PIM: " << LEN_PIM << std::endl;

	int n;
	if (option == 1)
		n = 65536;
	else if (option == 2)
		n = 131072;
	else if (option == 3)
		n = 4194304;
	else
		std::cout << "choose option in [1, 2, 3]\n";

	uint8_t *in0 = (uint8_t *)malloc(sizeof(uint16_t) * n);
	uint8_t *in1 = (uint8_t *)malloc(sizeof(uint16_t) * n);
	uint8_t *out = (uint8_t *)malloc(sizeof(uint16_t) * n);
	uint8_t *ans = (uint8_t *)malloc(sizeof(uint16_t) * n);

	for (int i = 0; i < n; i++)	{
		((uint16_t *)in0)[i] = rand();
		((uint16_t *)in1)[i] = rand();
		((uint16_t *)ans)[i] = ((uint16_t *)in0)[i] * ((uint16_t *)in1)[i];
	}

	if (DebugMode())
		std::cout << "///// Preprocessing MUL BLAS... /////\n";

	PIMKernel micro_kernel = PIMKernel();
	pimblasMulPreprocess(&micro_kernel, n, &in0, &in1);

	if (DebugMode())
		std::cout << "///// Testing MUL BLAS... /////\n";
	
	auto start = Time::now();
	pim_mul(micro_kernel, n, in0, in1, out);
	auto end = Time::now();
	fsec time = end - start;
	std::cout << time.count() << "s\n";


	if (DebugMode())
		std::cout << "///// Test MUL BLAS Ended!! /////\n";

	int error = 0;
	for (int i = 0; i < n; i++) {
		// std::cout << ((uint16_t *)out)[i] << " ";
		error = error + ABS(((uint16_t *)out)[i] - ((uint16_t *)ans)[i]);
	}
		
	std::cout << "\n\nERROR: " << error << std::endl;
	return;
}


void test_relu_blas(int option) {
	if (DebugMode())
		std::cout << "LEN_PIM: " << LEN_PIM << std::endl;

	int n;
	if (option == 1)
		n = 4096;
	else if (option == 2)
		n = 8192;
	else if (option == 3)
		n = 16384;
	else
		std::cout << "choose option in [1, 2, 3]\n";

	uint8_t *in = (uint8_t *)malloc(sizeof(uint16_t) * n);
	uint8_t *out = (uint8_t *)malloc(sizeof(uint16_t) * n);
	uint8_t *ans = (uint8_t *)malloc(sizeof(uint16_t) * n);

	for (int i = 0; i < n; i++)	{
		((uint16_t *)in)[i] = rand();
		((uint16_t *)ans)[i] = ReLU((uint16_t *)in)[i];
	}

	if (DebugMode()) 
		std::cout << "///// Preprocessing RELU BLAS... /////\n";

	PIMKernel micro_kernel = PIMKernel();
	pimblasReluPreprocess(&micro_kernel, n, &in);

	if (DebugMode())
		std::cout << "///// Testing RELU BLAS... /////\n";
	
	auto start = Time::now();
	pim_relu(micro_kernel, n, in, out);
	auto end = Time::now();
	fsec time = end - start;
	std::cout << time.count() << "s\n";


	if (DebugMode())
		std::cout << "///// Test RELU BLAS Ended!! /////\n";

	int error = 0;
	for (int i = 0; i < n; i++)	{
		//std::cout << ((uint16_t *)ans)[i] << " " << ((uint16_t *)out)[i] << std::endl;;
		// std::cout << ((uint16_t *)out)[i] << " ";
		error = error + ABS(((uint16_t *)out)[i] - ((uint16_t *)ans)[i]);
	}
		
	std::cout << "\n\nERROR: " << error << std::endl;
	return;
}

void test_mac_blas(int option) {
	if (DebugMode())
		std::cout << "LEN_PIM: " << LEN_PIM << std::endl;

	int n;
	if (option == 1)
		n = 65536;
	else if (option == 2)
		n = 131072;
	else if (option == 3)
		n = 4194304;
	else
		std::cout << "choose option in [1, 2, 3]\n";
}

void test_bn_blas(int option) {
	if (DebugMode())
		std::cout << "LEN_PIM: " << LEN_PIM << std::endl;
	int l;
	int f = 1024;

	if (option == 1)
		l = 64;
	else if (option == 2)
		l = 256;
	else if (option == 3)
		l = 576;
	else
		std::cout << "choose option in [1, 2, 3]\n";
	
	uint8_t *in = (uint8_t *)malloc(sizeof(uint16_t) * l * f);
	uint8_t *w0 = (uint8_t *)malloc(sizeof(uint16_t) * f);
	uint8_t *w1 = (uint8_t *)malloc(sizeof(uint16_t) * f);
	uint8_t *out = (uint8_t *)malloc(sizeof(uint16_t) * l * f);
	uint8_t *ans = (uint8_t *)malloc(sizeof(uint16_t) * l * f);

	for (int fi = 0; fi < f; fi++) {
		for (int li=0; li < l; li++)
			((uint16_t*)in)[li*f + fi] = rand();
		((uint16_t*)w0)[fi] = rand();
		((uint16_t*)w1)[fi] = rand();
	}

	for (int fi = 0; fi < f; fi++)
		for (int li=0; li < l; li++)
			((uint16_t*)ans)[li*f + fi] = ((uint16_t*)in)[li*f + fi] * ((uint16_t*)w0)[fi] + ((uint16_t*)w1)[fi];

	if (DebugMode())
		std::cout << "///// Preprocessing BN BLAS... /////\n";


	PIMKernel micro_kernel = PIMKernel();
	pimblasBn1dPreprocess(&micro_kernel, l, f, &w0, &w1);

	if (DebugMode())
		std::cout << "///// Testing BN BLAS... /////\n";

	auto start = Time::now();
	pim_bn1d(micro_kernel, l, f, in, w0, w1, out);
	auto end = Time::now();
	fsec time = end - start;
	std::cout << time.count() << "s\n";

	if (DebugMode())
		std::cout << "///// Test BN BLAS Ended!! /////\n";

	int error = 0;
	for (int li = 0; li < l; li++)
		for (int fi = 0; fi < f; fi++)
			error = error + ABS(((uint16_t *)out)[li*f + fi] - ((uint16_t *)ans)[li*f + fi]);
	
	std::cout << "\n\nERROR: " << error << std::endl;
	return;
}

void test_encoder_blas(){

	uint16_t in_token[68896];
	uint16_t *Wq;
	uint16_t *Wk;
	uint16_t *W;
	uint16_t output[1000];

	for(int i = 0; i < 68896; i++)
		in_token[i] = rand();

	uint32_t packed_data[100000];

	std::cout << "encoder execution start\n";

	// attention
	pimBlasAttentionPreprocess(256, 512, in_token, W, packed_data);
	auto start = Time::now();
	pim_attention(packed_data, 256, 512, output);
	auto end = Time::now();
	fsec time = end - start;
	std::cout << "attention time :" << time.count() << "\n";

	// layernorm at cpu

	// FFN
	pimBlasFFNPreprocess(packed_data, Wq, 256, 512);
	start = Time::now();
	pim_FFN(packed_data, 256, 512, output);
	end = Time::now();
	time = end - start;
	std::cout << "Feed Forward time :" << time.count() << "\n";
}

void test_gemv_blas(int option) {
	if (DebugMode())
		std::cout << "LEN_PIM: " << LEN_PIM << std::endl;
	int m;
	int n = 4096;
	if (option == 1)
		m = 512;
	else if (option == 2)
		m = 1024;
	else if (option == 3)
		m = 2048;
	else
		std::cout << "choose option in [1, 2, 3]\n";
	
	m = 512;
	n = 4096;

	uint8_t *in = (uint8_t *)malloc(sizeof(uint16_t) * m);
	uint8_t *w = (uint8_t *)malloc(sizeof(uint16_t) * m * n);
	uint8_t *out = (uint8_t *)malloc(sizeof(uint16_t) * n);
	uint8_t *ans = (uint8_t *)malloc(sizeof(uint16_t) * n);

	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			((uint16_t *)w)[i * m + j] = rand();

	for (int j = 0; j < m; j++)
		((uint16_t *)in)[j] = rand();

	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			((uint16_t *)ans)[i] = ((uint16_t *)ans)[i] + ((uint16_t *)w)[i * m + j] * ((uint16_t *)in)[j];

	if (DebugMode())
		std::cout << "///// Preprocessing GEMV BLAS... /////\n";

	struct PimCmdMetadata cmd_metadata0;
	struct PimCmdMetadata cmd_metadata1;
	pimblasGemvPreprocess(&cmd_metadata0, &cmd_metadata1, m, n, &w);

	if (DebugMode())
		std::cout << "///// Testing GEMV BLAS... /////\n";
	
	auto start = Time::now();
	pim_gemv(cmd_metadata0, cmd_metadata1, m, n, in, w, out);
	auto end = Time::now();
	fsec time = end - start;
	std::cout << time.count() << "s\n";


	if (DebugMode())
		std::cout << "///// Test GEMV BLAS Ended!! /////\n";

	if (DebugMode())
		std::cout << "///// Calculate Error /////\n";
	int error = 0;
	for (int i = 0; i < n; i++) {
		// std::cout << (int)((uint16_t *)out)[i] << " ";
		error = error + ABS(((uint16_t *)out)[i] - ((uint16_t *)ans)[i]);
	}
		
	std::cout << "ERROR: " << error << std::endl;
	return;
}

void test_lstm_blas(int option) {
	if (DebugMode())
		std::cout << "LEN_PIM: " << LEN_PIM << std::endl;
	
	int m = 640;
	int n = 4096;

	uint8_t *in = (uint8_t *)malloc(sizeof(uint16_t) * m);
	uint8_t *w = (uint8_t *)malloc(sizeof(uint16_t) * m * n);
	uint8_t *b = (uint8_t *)malloc(sizeof(uint16_t) * n);
	uint8_t *out = (uint8_t *)malloc(sizeof(uint16_t) * n);
	uint8_t *ans = (uint8_t *)malloc(sizeof(uint16_t) * n);

	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			((uint16_t *)w)[i * m + j] = rand();

	for (int j = 0; j < m; j++)
		((uint16_t *)in)[j] = rand();

	for (int i = 0; i < n; i++)
		((uint16_t *)b)[i] = rand();

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++)
			((uint16_t *)ans)[i] = ((uint16_t *)ans)[i] + ((uint16_t *)w)[i * m + j] * ((uint16_t *)in)[j];
		((uint16_t *)ans)[i] = ((uint16_t*)ans)[i] + ((uint16_t*)b)[i];
	}

	if (DebugMode())
		std::cout << "///// Preprocessing LSTM BLAS... /////\n";

	PIMKernel micro_kernel = PIMKernel();
	pimblasLstmPreprocess(&micro_kernel, m, n, &w, &b);

	if (DebugMode())
		std::cout << "///// Testing LSTM BLAS... /////\n";
	
	auto start = Time::now();
	pim_lstm(micro_kernel, m, n, in, w, b, out);
	auto end = Time::now();
	fsec time = end - start;
	std::cout << time.count() << "s\n";


	if (DebugMode())
		std::cout << "///// Test LSTM BLAS Ended!! /////\n";

	if (DebugMode())
		std::cout << "///// Calculate Error /////\n";
	int error = 0;
	for (int i = 0; i < n; i++) {
		// std::cout << (int)((uint16_t *)out)[i] << " ";
		error = error + ABS(((uint16_t *)out)[i] - ((uint16_t *)ans)[i]);
	}
		
	std::cout << "ERROR: " << error << std::endl;
	return;
}

int main(int argc, char **argv) {
	int option;
	std::cout << "option : 1 / 2 / 3\nenter option :";
	std::cin >> option;

	blas_init(0);

	if (argc <= 1) {
		if (DebugMode())
			std::cout << "add, mul, mac, bn, gemv, lstm\n";
		return -1;
	}

	if (std::string(argv[1]) == "add")
		test_add_blas(option);
	else if (std::string(argv[1]) == "mul")
		test_mul_blas(option);
	else if (std::string(argv[1]) == "mac")
		test_mac_blas(option);
	else if (std::string(argv[1]) == "relu")
		test_relu_blas(option);
	else if (std::string(argv[1]) == "bn")
		test_bn_blas(option);
	else if (std::string(argv[1]) == "gemv")
		test_gemv_blas(option);
	else if (std::string(argv[1]) == "lstm")
		test_lstm_blas(option);
	else if (std::string(argv[1]) == "encoder")
		test_encoder_blas();
	else
		std::cout << "error... not supported\n";
	return 0;
}


