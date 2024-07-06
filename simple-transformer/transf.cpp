#include <ATen/ATen.h>
#include <iostream>

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
            ctx->save_for_backward({
                input,
                WQ, BQ,
                WK, BK,
                WV, BV, 
                WX, BX,
                WF1, BF1,
                WF2, BF2
            });
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