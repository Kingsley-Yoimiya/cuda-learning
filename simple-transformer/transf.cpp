#include <ATen/ATen.h>
#include <iostream>
#include <math.h>
#include <tuple>

const double eps = 1e-6;

class transformerFunc : public torch::autograd::Function<transformerFunc> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx, 
        torch::Tensor input, int d_k,
        torch::Tensor WQ, torch::Tensor BQ,
        torch::Tensor WK, torch::Tensor BK,
        torch::Tensor WV, torch::Tensor BV, 
        torch::Tensor WX, torch::Tensor BX,
        torch::Tensor WF1, torch::Tensor BF1,
        torch::Tensor WF2, torch::Tensor BF2) {
            int B = input.size(0), L = input.size(1), D = input.size(2);
            ctx->save_for_backward({
                input, WQ, BQ,
                WK, BK,
                WV, BV, 
                WX, BX,
                WF1, BF1,
                WF2, BF2
            });
            ctx->saved_data["d_k"] = d_k;
            auto Q = (torch::matmul(input, WQ) + BQ).view({B, -1, L, d_k}).transpose(1, 2);
            auto KT = (torch::matmul(input, WK) + BK).view({B, -1, L, d_k}).transpose(1, 2).transpose(-2, -1);
            auto V = (torch::matmul(input, WV) + BV).view({B, -1, L, d_k}).transpose(1, 2);
            auto scores = torch::matmul(Q, KT) / sqrt(D / d_k);
            auto output = torch::matmul(scores.softmax(-1), V);
            output = output.transpose(1, 2).contiguous().view({ B, L, D }) + input;
            output = torch::matmul(output, WX) + BX;
            // LayerNorm
            auto mean = output.mean(-1, true);
            auto std = output.std(-1, true, true); // refer to https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6TensorStEN2at11DimnameListEbb
            // printf("%d %d\n", mean.dim(), std.dim());
            // printf("%d %d\n", mean.size(0), mean.size(1));
            // printf("%d %d\n", std.size(0), std.size(1));
            output = (output - mean) / (std + eps);

            output = (output - mean) / (std + eps); // output1 
            output = (torch::matmul((torch::matmul(output, WF1) + BF1), WF2) + BF2) + output;
            // temp1: (torch::matmul(output, WF1) + BF1)
            // LayerNorm
            mean = output.mean(-1, true);
            std = output.std(-1, true, true);
            output = (output - mean) / (std + eps);

            
            // TORCH_CHECK(output.dim() == 3, "ans' dim = 3");
            return input;
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_output) {
        auto saved = ctx->get_saved_variables();
        torch::Tensor input = saved[0];
        torch::Tensor WQ = saved[1];
        torch::Tensor BQ = saved[2];
        torch::Tensor WK = saved[3];
        torch::Tensor BK = saved[4];
        torch::Tensor WV = saved[5];
        torch::Tensor BV = saved[6];
        torch::Tensor WX = saved[7];
        torch::Tensor BX = saved[8];
        torch::Tensor WF1 = saved[9];
        torch::Tensor BF1 = saved[10];
        torch::Tensor WF2 = saved[11];
        torch::Tensor BF2 = saved[12];
        return {
            input, torch::Tensor(),
            WQ, BQ,
            WK, BK,
            WV, BV, 
            WX, BX,
            WF1, BF1,
            WF2, BF2
        };
    }
};

torch::Tensor forward(
    torch::Tensor input, int d_k,
    torch::Tensor WQ, torch::Tensor BQ,
    torch::Tensor WK, torch::Tensor BK,
    torch::Tensor WV, torch::Tensor BV, 
    torch::Tensor WX, torch::Tensor BX,
    torch::Tensor WF1, torch::Tensor BF1,
    torch::Tensor WF2, torch::Tensor BF2) {
    return transformerFunc::apply(
        input, d_k,
        WQ, BQ,
        WK, BK,
        WV, BV, 
        WX, BX,
        WF1, BF1,
        WF2, BF2
    );
}