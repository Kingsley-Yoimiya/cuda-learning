#include <ATen/ATen.h>
#include <iostream>
#include <math.h>
#include <tuple>
#include <vector>

const double eps = 1e-6;

// d_k means heads num

class transformerFunc : public torch::autograd::Function<transformerFunc> {
public:
  static torch::Tensor
  forward(torch::autograd::AutogradContext *ctx, torch::Tensor input, int d_k,
          torch::Tensor WQ, torch::Tensor BQ, torch::Tensor WK,
          torch::Tensor BK, torch::Tensor WV, torch::Tensor BV,
          torch::Tensor WX, torch::Tensor BX, torch::Tensor WF1,
          torch::Tensor BF1, torch::Tensor WF2, torch::Tensor BF2) {
    int B = input.size(0), L = input.size(1), D = input.size(2);
    std::vector<torch::Tensor> saved_list = {input, WQ, BQ,  WK,  BK,  WV, BV,
                                             WX,    BX, WF1, BF1, WF2, BF2};
    ctx->saved_data["d_k"] = d_k;
    auto Q = (torch::matmul(input, WQ) + BQ)
                 .view({B, -1, L / d_k, D})
                 .transpose(1, 2);
    auto KT = (torch::matmul(input, WK) + BK)
                  .view({B, -1, L / d_k, D})
                  .transpose(1, 2)
                  .transpose(-2, -1);
    auto V = (torch::matmul(input, WV) + BV)
                 .view({B, -1, L / d_k, D})
                 .transpose(1, 2);
    auto scores = torch::matmul(Q, KT) / sqrt(D);
    auto output = torch::matmul(scores.softmax(-1), V);
    output = output.transpose(1, 2).contiguous().view({B, L, D}); // output2
    saved_list.emplace_back(output);
    output = torch::matmul(output, WX) + BX + input;
    // LayerNorm
    auto mean = output.mean(-1, true);
    auto std = output.std(-1, false, true);
    // refer to
    // https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6TensorStEN2at11DimnameListEbb
    saved_list.emplace_back(mean);          // mean1
    saved_list.emplace_back(std);           // std1
    saved_list.emplace_back(output);        // output_t1
    output = (output - mean) / (std + eps); // output1
    saved_list.emplace_back(output);
    output =
        (torch::matmul((torch::matmul(output, WF1) + BF1), WF2) + BF2) + output;
    mean = output.mean(-1, true);
    std = output.std(-1, false, true);
    saved_list.emplace_back(mean);   // mean2
    saved_list.emplace_back(std);    // std2
    saved_list.emplace_back(output); // output_t2
    output = (output - mean) / (std + eps);
    ctx->save_for_backward(saved_list);
    return output;
  }

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::tensor_list grad_output) {
    auto device = torch::cuda::is_available() ? torch::Device("cuda:0")
                                              : torch::Device("cpu");
    auto saved = ctx->get_saved_variables();
    auto d_output = grad_output[0];
    int B = saved[0].size(0), L = saved[0].size(1), D = saved[0].size(2);
    int d_k = ctx->saved_data["d_k"].toInt(), SK = saved[9].size(1);
    torch::Tensor input = saved[0], d_input = torch::zeros({B, L, D}, device),
                  WQ = saved[1], d_WQ = torch::zeros({D, D}, device),
                  BQ = saved[2], d_BQ = torch::zeros({D}, device),
                  WK = saved[3], d_WK = torch::zeros({D, D}, device),
                  BK = saved[4], d_BK = torch::zeros({D}, device),
                  WV = saved[5], d_WV = torch::zeros({D, D}, device),
                  BV = saved[6], d_BV = torch::zeros({D}, device),
                  WX = saved[7], d_WX = torch::zeros({D, D}, device),
                  BX = saved[8], d_BX = torch::zeros({D}, device),
                  WF1 = saved[9], d_WF1 = torch::zeros({D, SK}, device),
                  BF1 = saved[10], d_BF1 = torch::zeros({SK}, device),
                  WF2 = saved[11], d_WF2 = torch::zeros({SK, D}, device),
                  BF2 = saved[12], d_BF2 = torch::zeros({D}, device),
                  output2 = saved[13], mean1 = saved[14], std1 = saved[15],
                  output_t1 = saved[16], output1 = saved[17], mean2 = saved[18],
                  std2 = saved[19], output_t2 = saved[20];

    auto batchnorm_backward = [&](torch::Tensor &dL_doutput_norm,
                                  const torch::Tensor &output,
                                  const torch::Tensor &mean,
                                  const torch::Tensor &std) {
      auto sigma = std + eps;
      auto dsigma = torch::sum(
          dL_doutput_norm * (output - mean) * -1 / (sigma * sigma), -1, true);
      auto dmean = torch::sum(dL_doutput_norm * -1 / sigma, -1, true) -
                   dsigma / sigma * (output - mean).mean(-1, true);
      dL_doutput_norm = dL_doutput_norm / sigma +
                        dsigma * (output - mean) / D / sigma + dmean / D;
      return dL_doutput_norm;
    };

    batchnorm_backward(d_output, output_t2, mean2, std2);

    d_BF2 = d_output.sum(at::IntArrayRef({0, 1}));
    d_WF2 = torch::matmul((torch::matmul(output1, WF1) + BF1).transpose(-2, -1),
                          d_output)
                .sum(0);
    auto tmp = torch::matmul(d_output, WF2.transpose(-2, -1));
    d_BF1 = tmp.sum(at::IntArrayRef({0, 1}));
    d_WF1 = torch::matmul(output1.transpose(-2, -1), tmp).sum(0);
    d_output = torch::matmul(tmp, WF1.transpose(-2, -1)) + d_output;

    batchnorm_backward(d_output, output_t1, mean1, std1);
    d_input += d_output;
    d_BX = d_output.sum(at::IntArrayRef({0, 1}));
    d_WX = torch::matmul(output2.transpose(-2, -1), d_output).sum(0);
    d_output = torch::matmul(d_output, WX.transpose(-2, -1));
    d_output = d_output.contiguous().view({B, -1, L / d_k, D}).transpose(1, 2);
    auto Q = (torch::matmul(input, WQ) + BQ)
                 .view({B, -1, L / d_k, D})
                 .transpose(1, 2);
    auto K = (torch::matmul(input, WK) + BK)
                 .view({B, -1, L / d_k, D})
                 .transpose(1, 2);
    auto V = (torch::matmul(input, WV) + BV)
                 .view({B, -1, L / d_k, D})
                 .transpose(1, 2);
    auto scores_sm =
        (torch::matmul(Q, K.transpose(-2, -1)) / sqrt(D)).softmax(-1);
    auto d_V = torch::matmul(scores_sm.transpose(-2, -1), d_output)
                   .transpose(1, 2)
                   .contiguous()
                   .view({B, L, D});
    d_output = torch::matmul(d_output, V.transpose(-2, -1));
    // d_softmax !
    d_output = scores_sm * (d_output - (scores_sm * d_output).sum({-1}, true));
    //
    auto d_Q = torch::matmul(d_output, K / sqrt(D))
                   .transpose(1, 2)
                   .contiguous()
                   .view({B, L, D});
    auto d_K = torch::matmul(Q.transpose(-2, -1) / sqrt(D), d_output)
                   .transpose(-2, -1)
                   .transpose(1, 2)
                   .contiguous()
                   .view({B, L, D});
    // puts("PASSED #7");
    // notice: d_* means the (input \times W*) + B*, except d_R
    auto linearBackward = [&](const torch::Tensor &dy, torch::Tensor &dw,
                              torch::Tensor &db, const torch::Tensor &W) {
      // y (B, L, D), W(D, D), B(D)
      // y = input * W + B
      db = dy.sum(at::IntArrayRef({0, 1}));
      dw = torch::matmul(input.transpose(-2, -1), dy).sum(0);
      d_input += torch::matmul(dy, W.transpose(-2, -1));
    };
    linearBackward(d_V, d_WV, d_BV, WV);
    linearBackward(d_Q, d_WQ, d_BQ, WQ);
    linearBackward(d_K, d_WK, d_BK, WK);
    return {
        d_input, torch::Tensor(), d_WQ,  d_BQ,  d_WK, d_BK, d_WV, d_BV, d_WX,
        d_BX,    d_WF1,           d_BF1, d_WF2, d_BF2};
  }
};

torch::Tensor forward(torch::Tensor input, int d_k, torch::Tensor WQ,
                      torch::Tensor BQ, torch::Tensor WK, torch::Tensor BK,
                      torch::Tensor WV, torch::Tensor BV, torch::Tensor WX,
                      torch::Tensor BX, torch::Tensor WF1, torch::Tensor BF1,
                      torch::Tensor WF2, torch::Tensor BF2) {
  return transformerFunc::apply(input, d_k, WQ, BQ, WK, BK, WV, BV, WX, BX, WF1,
                                BF1, WF2, BF2);
}