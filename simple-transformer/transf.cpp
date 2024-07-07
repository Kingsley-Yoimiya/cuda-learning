#include <ATen/ATen.h>
#include <iostream>
#include <math.h>
#include <tuple>
#include <vector>

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
            std::vector<torch::Tensor> saved_list = {
                input, WQ, BQ,
                WK, BK,
                WV, BV, 
                WX, BX,
                WF1, BF1,
                WF2, BF2
            };
            ctx->saved_data["d_k"] = d_k;
            auto Q = (torch::matmul(input, WQ) + BQ).view({B, -1, L, D / d_k}).transpose(1, 2);
            auto KT = (torch::matmul(input, WK) + BK).view({B, -1, L, D / d_k}).transpose(1, 2).transpose(-2, -1);
            auto V = (torch::matmul(input, WV) + BV).view({B, -1, L, D / d_k}).transpose(1, 2);
            auto scores = torch::matmul(Q, KT) / sqrt(D / d_k);
            auto output = torch::matmul(scores/*.softmax(-1)*/, V);
            output = output.transpose(1, 2).contiguous().view({ B, L, D }) + input; // output2
            saved_list.emplace_back(output);
            output = torch::matmul(output, WX) + BX;
            // LayerNorm
            auto mean = output.mean(-1, true);
            auto std = output.std(-1, true, true); // refer to https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6TensorStEN2at11DimnameListEbb
            saved_list.emplace_back(std); // std1
            // printf("%d %d\n", mean.dim(), std.dim());
            // printf("%d %d\n", mean.size(0), mean.size(1));
            // printf("%d %d\n", std.size(0), std.size(1));
            output = (output - mean) / (std + eps); // output1 
            saved_list.emplace_back(output);
            output = (torch::matmul((torch::matmul(output, WF1) + BF1), WF2) + BF2) + output;
            // temp1: (torch::matmul(output, WF1) + BF1)
            // temp2: (torch::matmul(output.transpose(1, 2), d_output, WF2.transpose(-2, -1)))
            // LayerNorm
            mean = output.mean(-1, true);
            std = output.std(-1, true, true);
            output = (output - mean) / (std + eps);
            saved_list.emplace_back(std); // std2
            
            ctx->save_for_backward(saved_list);
            // TORCH_CHECK(output.dim() == 3, "ans' dim = 3");
            return output;
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_output) {
        auto saved = ctx->get_saved_variables();
        // printf("!LEN: %d\n", saved.size());
        auto d_output = grad_output[0];
        int B = saved[0].size(0), L = saved[0].size(1), D = saved[0].size(2);
        int d_k = ctx->saved_data["d_k"].toInt();
        // printf("%d %d %d %d\n", B, L, D, d_k);
        torch::Tensor input = saved[0], d_input = torch::empty({ B, L, D });
        // puts("PASSED #0");
        torch::Tensor WQ = saved[1], d_WQ = torch::empty({ D, D });
        // puts("PASSED #0");
        torch::Tensor BQ = saved[2], d_BQ = torch::empty({ D });
        torch::Tensor WK = saved[3], d_WK = torch::empty({ D, D });
        torch::Tensor BK = saved[4], d_BK = torch::empty({ D });
        torch::Tensor WV = saved[5], d_WV = torch::empty({ D, D });
        torch::Tensor BV = saved[6], d_BV = torch::empty({ D });
        // puts("PASSED #0");
        torch::Tensor WX = saved[7], d_WX = torch::empty({ D, D });
        torch::Tensor BX = saved[8], d_BX = torch::empty({ D });
        
        int SK = saved[9].size(1);
        torch::Tensor WF1 = saved[9], d_WF1 = torch::empty({ D, SK });
        torch::Tensor BF1 = saved[10], d_BF1 = torch::empty({ SK });
        torch::Tensor WF2 = saved[11], d_WF2 = torch::empty({ SK, D });
        torch::Tensor BF2 = saved[12], d_BF2 = torch::empty({ D });
        torch::Tensor output2 = saved[13];
        torch::Tensor std1 = saved[14];
        torch::Tensor output1 = saved[15];
        torch::Tensor std2 = saved[16];
        // puts("PASSED #0");
        d_output /= (std2 + eps);

        d_BF2 = d_output.sum(at::IntArrayRef({0, 1}));
        d_WF2 = torch::matmul((torch::matmul(output1, WF1) + BF1).transpose(-2, -1), d_output).sum(0);
        // puts("PASSED #1");
        auto tmp = torch::matmul(d_output, WF2.transpose(-2, -1));
        d_BF1 = tmp.sum(at::IntArrayRef({0, 1}));
        // puts("PASSED #2");
        d_WF1 = torch::matmul(output1.transpose(-2, -1), tmp).sum(0);
        d_output = torch::matmul(tmp, WF1.transpose(-2, -1)) + d_output;
        // puts("PASSED #3");
        
        d_output /= (std1 + eps);

        d_BX = d_output.sum(at::IntArrayRef({0, 1}));
        d_WX = torch::matmul(output2.transpose(-2, -1), d_output).sum(0);
        d_output = torch::matmul(d_output, WX.transpose(-2, -1));
        // puts("PASSED #4");
        d_input += d_output;
        d_output = d_output.contiguous().view({B, -1, L, D / d_k}).transpose(1, 2);
        // puts("PASSED #5");
        // notice: the generation of Q, KT need scale. 
        auto Q = (torch::matmul(input, WQ) + BQ).view({B, -1, L, D / d_k}).transpose(1, 2) / sqrt(D / d_k);
        auto K = (torch::matmul(input, WK) + BK).view({B, -1, L, D / d_k}).transpose(1, 2) / sqrt(D / d_k);
        auto V = (torch::matmul(input, WV) + BV).view({B, -1, L, D / d_k}).transpose(1, 2);
        auto scores_sm = (torch::matmul(Q, K.transpose(-2, -1)) * sqrt(D / d_k));//.softmax(-1);
        //auto scores_sm = scores.softmax(-1);
        // puts("PASSED #6");
        auto d_V = torch::matmul(scores_sm.transpose(-2, -1), d_output).transpose(1, 2).contiguous().view({B, L, D});
        d_output = torch::matmul(d_output, V.transpose(-2, -1));
        // d_softmax !
        //printf("%d %d %d\n", d_R.size(0), d_R.size(1), d_R.size(2));
        // d_R = scores_sm * (d_R - (scores_sm * d_R).sum({-1}, true));
        // 
        auto d_Q = torch::matmul(d_output, K).transpose(1, 2).contiguous().view({B, L, D});
        auto d_K = torch::matmul(Q.transpose(-2, -1), d_output).transpose(-2, -1).transpose(1, 2).contiguous().view({B, L, D}); 
        // puts("PASSED #7");
        // notice: d_* means the (input \times W*) + B*, except d_R
        auto linearBackward = [&](const torch::Tensor &dy, torch::Tensor &dw, torch::Tensor &db, const torch::Tensor &W) {
            // y (B, L, D), W(D, D), B(D)
            // y = input * W + B
            // printf("%d %d %d\n", dy.size(0), dy.size(1), dy.size(2));
            // printf("%d %d\n", W.size(0), W.size(1));
            db = dy.sum(at::IntArrayRef({0, 1}));
            dw = torch::matmul(input.transpose(-2, -1), dy).sum(0);
            d_input += torch::matmul(dy, W.transpose(-2, -1));
        };
        linearBackward(d_V, d_WV, d_BV, WV);
        linearBackward(d_Q, d_WQ, d_BQ, WQ);
        linearBackward(d_K, d_WK, d_BK, WK);
        // puts("PASSED #8");
        // puts("GENER!");
        return {
            d_input, torch::Tensor(),
            d_WQ, d_BQ,
            d_WK, d_BK,
            d_WV, d_BV, 
            d_WX, d_BX,
            d_WF1, d_BF1,
            d_WF2, d_BF2
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