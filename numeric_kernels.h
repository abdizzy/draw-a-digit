/*
 * numeric_kernels.h - small dense numeric kernels for C.
 *
 * All routines are single-precision (float) and row-major.
 * Output buffers must not alias the input buffers unless the function
 * is documented as in-place.
 *
 * Build with: cc -std=c11 -O2 -I. your_app.c numeric_kernels.c -lm
 */
#ifndef NUMERIC_KERNELS_H
#define NUMERIC_KERNELS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* C = A * B: A is m x k, B is k x n, C is m x n (row-major). */
void nk_matmul(const float *A, const float *B, float *C,
               size_t m, size_t n, size_t k);

/* y[i] = max(0, x[i]) */
void nk_relu(const float *x, float *y, size_t n);

/* y[i] = x[i] if x[i] > 0, else slope * x[i] */
void nk_leaky_relu(const float *x, float *y, size_t n, float slope);

/* y[i] = 1 / (1 + exp(-x[i])) */
void nk_sigmoid(const float *x, float *y, size_t n);

/* y[i] = tanh(x[i]) */
void nk_tanh(const float *x, float *y, size_t n);

/* y[i] = exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))) */
void nk_gelu(const float *x, float *y, size_t n);

/* y[i] = tanh approximation of GELU */
void nk_gelu_tanh(const float *x, float *y, size_t n);

/* In-place numerically stable softmax over x[0..n). */
void nk_softmax_inplace(float *x, size_t n);

/* y = softmax(x); y and x may point to the same buffer. */
void nk_softmax(const float *x, float *y, size_t n);

/* In-place numerically stable softmax on each of the rows rows x cols. */
void nk_softmax_rows(float *x, size_t rows, size_t cols);

#ifdef __cplusplus
}
#endif

#endif /* NUMERIC_KERNELS_H */
