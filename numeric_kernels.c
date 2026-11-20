#include "numeric_kernels.h"

#include <math.h>

void nk_matmul(const float *A, const float *B, float *C,
               size_t m, size_t n, size_t k)
{
    for (size_t i = 0; i < m; i++) {
        float *c = C + i * n;
        for (size_t j = 0; j < n; j++)
            c[j] = 0.0f;
        for (size_t p = 0; p < k; p++) {
            const float a = A[i * k + p];
            if (a == 0.0f)
                continue;
            const float *b = B + p * n;
            for (size_t j = 0; j < n; j++)
                c[j] += a * b[j];
        }
    }
}

void nk_relu(const float *x, float *y, size_t n)
{
    for (size_t i = 0; i < n; i++)
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

void nk_leaky_relu(const float *x, float *y, size_t n, float slope)
{
    for (size_t i = 0; i < n; i++)
        y[i] = x[i] > 0.0f ? x[i] : slope * x[i];
}

void nk_sigmoid(const float *x, float *y, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        const float v = x[i];
        if (v >= 0.0f) {
            y[i] = 1.0f / (1.0f + expf(-v));
        } else {
            /* Avoids overflow of expf for large negative v. */
            const float e = expf(v);
            y[i] = e / (1.0f + e);
        }
    }
}

void nk_tanh(const float *x, float *y, size_t n)
{
    for (size_t i = 0; i < n; i++)
        y[i] = tanhf(x[i]);
}

void nk_gelu(const float *x, float *y, size_t n)
{
    const float c = 0.70710678118654752440f; /* 1/sqrt(2) */
    for (size_t i = 0; i < n; i++)
        y[i] = 0.5f * x[i] * (1.0f + erff(x[i] * c));
}

void nk_gelu_tanh(const float *x, float *y, size_t n)
{
    const float c = 0.79788456080286535588f; /* sqrt(2/pi) */
    const float a = 0.044715f;
    for (size_t i = 0; i < n; i++) {
        const float v = x[i];
        y[i] = 0.5f * v * (1.0f + tanhf(c * (v + a * v * v * v)));
    }
}

static void softmax_core(float *v, size_t n)
{
    if (n == 0)
        return;
    float m = v[0];
    for (size_t i = 1; i < n; i++)
        if (v[i] > m)
            m = v[i];
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        v[i] = expf(v[i] - m);
        sum += v[i];
    }
    if (sum > 0.0f) {
        const float inv = 1.0f / sum;
        for (size_t i = 0; i < n; i++)
            v[i] *= inv;
    } else {
        /* Degenerate input (e.g. all -inf): fall back to uniform. */
        const float inv = 1.0f / (float)n;
        for (size_t i = 0; i < n; i++)
            v[i] = inv;
    }
}

void nk_softmax_inplace(float *x, size_t n)
{
    softmax_core(x, n);
}

void nk_softmax(const float *x, float *y, size_t n)
{
    for (size_t i = 0; i < n; i++)
        y[i] = x[i];
    softmax_core(y, n);
}

void nk_softmax_rows(float *x, size_t rows, size_t cols)
{
    for (size_t r = 0; r < rows; r++)
        softmax_core(x + r * cols, cols);
}
