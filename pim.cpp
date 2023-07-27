#include <torch/extension.h>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <pybind11/pybind11.h>
#include "pim_config.h"
#include "pim_blas.h"
#include <torch/torch.h>
#include <iostream>

namespace py = pybind11;

void blas_init(uint64_t pim_base);
//bool pim_gemv(PIMKernel micro_kernel, int len_in, int len_out, uint8_t *in, uint8_t *weight, uint8_t *out);

bool isSuitableOps(PIM_OP op, int len0, int len1);
PIMKernel GetMicrokernelCode(PIM_OP op, PIM_OP_ATTRS op_attrs);
uint8_t *MapMemory(uint8_t *data, size_t len);

struct d_ptr
{
    uint8_t *data_ptr;
};

d_ptr MapMemory_(torch::Tensor T, PIMKernel micro_kernel, size_t len)
{
    if (micro_kernel.layout == 1)
        T = T.transpose(0, 1);
    uint8_t *data = (uint8_t *)T.data_ptr();
    d_ptr tmp;
    tmp.data_ptr = MapMemory(data, len);
    return tmp;
}

torch::Tensor pim_linear_forward(
    PIMKernel micro_kernel,
    torch::Tensor input,
    d_ptr weight_ptr,
    int len_in,
    int len_out)
{
    std::cout << "PIM Linear! \n";
    uint8_t *in = (uint8_t *)input.data_ptr();
    uint8_t *out = (uint8_t *)malloc(sizeof(uint16_t) * len_out);
    uint8_t *weight = weight_ptr.data_ptr;
    //pim_gemv(micro_kernel, len_in, len_out, in, weight, out);
    torch::Tensor out_tensor = torch::from_blob(out, {len_out});
    return out_tensor;
}

//torch::Tensor pim_msa(torch::Tensor input, torch::Tensor weight, int dim, int len)
/*torch::Tensor pim_msa(torch::Tensor input, torch::Tensor weight, int seq_len, int emb_dim)
{
    // std::cout << input << std::endl;
    // std::cout << weight << std::endl;
    bool tmp;
    // int emb_dim = output.size(1);
    // int seq_len = output.size(0);
    // uint16_t *out = (uint16_t *)output.data_ptr();

    uint16_t *in = (uint16_t *)input.data_ptr();
    uint16_t *w = (uint16_t *)weight.data_ptr();
    uint32_t packed_data[100000];
    // printf("output size %u", sizeof(out));
    printf("emb_dim :%d\n", emb_dim);
    printf("seq_len :%d\n", seq_len);

    uint16_t *out = (uint16_t *)malloc(sizeof(uint16_t) * seq_len * emb_dim);
    pimBlasAttentionPreprocess(seq_len, emb_dim, in, w, packed_data);
    tmp = pim_attention(packed_data, seq_len, emb_dim, out);
    if(tmp == true)
        printf("true\n");
    else 
        printf("flase\n");

    // for(int i = 0; i < 5; i++){
    //     printf("%u ", output[i]);
    // }
    torch::Tensor out_tensor = torch::from_blob(out, {emb_dim * seq_len});

    return out_tensor;
}*/

torch::Tensor pim_msa(torch::Tensor input, torch::Tensor weight,torch::Tensor ffnw1,torch::Tensor ffnw2,  int seq_len, int emb_dim)
{
    bool tmp;
    // int emb_dim = output.size(1);
    // int seq_len = output.size(0);
    // uint16_t *out = (uint16_t *)output.data_ptr();

    uint16_t *in = (uint16_t *)input.data_ptr();
    uint16_t *w = (uint16_t *)weight.data_ptr();
    uint16_t *w1 = (uint16_t *)ffnw1.data_ptr();
    uint16_t *w2 = (uint16_t *)ffnw2.data_ptr();
    uint32_t packed_data[100000];
    printf("emb_dim :%d\n", emb_dim);
    printf("seq_len :%d\n", seq_len);

    uint16_t *out = (uint16_t *)malloc(sizeof(uint16_t) * seq_len * emb_dim);
    pimBlasAttentionPreprocess(seq_len, emb_dim, in, w, packed_data);
    tmp = pim_attention(packed_data, seq_len, emb_dim, out);
    if(tmp == true)
        printf("true\n");
    else 
        printf("flase\n");

    torch::Tensor out_tensor = torch::from_blob(out, {emb_dim * seq_len});

    return out_tensor;
}

torch::Tensor pim_ffn(torch::Tensor input, int seq_len, int emb_dim)
{
    // std::cout << input << std::endl;
    // std::cout << weight << std::endl;
    bool tmp;
    // int emb_dim = output.size(1);
    // int seq_len = output.size(0);
    // uint16_t *out = (uint16_t *)output.data_ptr();

    uint16_t *in = (uint16_t *)input.data_ptr();


    uint16_t *out = (uint16_t *)malloc(sizeof(uint16_t) * seq_len * emb_dim);
    //tmp = pim_FFN(packed_data, seq_len, emb_dim, out);
    std::cout << "seq_len : " << seq_len << std::endl;
    std::cout << "emb_dim : " << emb_dim << std::endl;
    torch::Tensor out_tensor = torch::from_blob(in, {seq_len, emb_dim});

    return out_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("GetMicrokernelCode", &GetMicrokernelCode, "Compile & Return MicrokernelCode");
    m.def("isSuitableOps", &isSuitableOps, "Check if it fits to PIM");
    m.def("linear_forward", &pim_linear_forward, "PIM Linear forward");
    m.def("blas_init", &blas_init, "initialize pim");
    m.def("MapMemory", &MapMemory_, "MapMemory");
    m.def("pim_msa", &pim_msa, "pim_msa");
    m.def("pim_ffn", &pim_ffn, "pim_ffn");
    py::class_<d_ptr>(m, "d_ptr")
        .def(py::init());

    py::enum_<PIM_OP>(m, "PIM_OP", py::arithmetic())
        .value("ADD", PIM_OP::ADD)
        .value("MUL", PIM_OP::MUL)
        .value("BN", PIM_OP::BN)
        .value("GEMV", PIM_OP::GEMV)
        .value("LSTM", PIM_OP::LSTM)
        .value("RELU", PIM_OP::RELU);

    py::class_<PIM_OP_ATTRS>(m, "PIM_OP_ATTRS")
        .def(py::init())
        .def("ADD", &PIM_OP_ATTRS::ADD)
        .def("MUL", &PIM_OP_ATTRS::MUL)
        .def("BN", &PIM_OP_ATTRS::BN)
        .def("GEMV", &PIM_OP_ATTRS::GEMV)
        .def("LSTM", &PIM_OP_ATTRS::LSTM);

    py::class_<PIMKernel>(m, "PIMKernel")
        .def("SetMicrokernelCode", &PIMKernel::SetMicrokernelCode);
}
