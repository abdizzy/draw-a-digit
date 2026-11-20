#include "numeric_kernels.h"

#include <math.h>
#include <stdio.h>

static int failures = 0;

static int near(float a, float b, float eps)
{
    return fabsf(a - b) <= eps;
}

#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
            failures++;                                                 \
        }                                                               \
    } while (0)

static void test_matmul(void)
{
    /* A is 2x3, B is 3x2: expect [58 64; 139 154]. */
    float A[6] = { 1, 2, 3, 4, 5, 6 };
    float B[6] = { 7, 8, 9, 10, 11, 12 };
    float C[4] = { 0 };
    nk_matmul(A, B, C, 2, 2, 3);
    CHECK(near(C[0], 58.0f, 1e-4f));
    CHECK(near(C[1], 64.0f, 1e-4f));
    CHECK(near(C[2], 139.0f, 1e-4f));
    CHECK(near(C[3], 154.0f, 1e-4f));

    /* A (3x3) * I (3x3) == A. */
    float M[9] = { 2, 0, 0, 0, 3, 0, 0, 0, 4 };
    float I[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    float R[9] = { 0 };
    nk_matmul(M, I, R, 3, 3, 3);
    CHECK(near(R[0], 2.0f, 1e-6f));
    CHECK(near(R[4], 3.0f, 1e-6f));
    CHECK(near(R[8], 4.0f, 1e-6f));

    /* k == 0 gives zeros. */
    float Z[4] = { 9, 9, 9, 9 };
    nk_matmul(A, B, Z, 2, 2, 0);
    CHECK(Z[0] == 0.0f && Z[1] == 0.0f && Z[2] == 0.0f && Z[3] == 0.0f);
}

static void test_activations(void)
{
    float x[4] = { -1.0f, 0.0f, 2.0f, -3.0f };
    float y[4];

    nk_relu(x, y, 4);
    CHECK(near(y[0], 0.0f, 0.0f) && near(y[1], 0.0f, 0.0f));
    CHECK(near(y[2], 2.0f, 0.0f) && near(y[3], 0.0f, 0.0f));

    nk_leaky_relu(x, y, 4, 0.01f);
    CHECK(near(y[0], -0.01f, 1e-7f) && near(y[1], 0.0f, 0.0f));
    CHECK(near(y[2], 2.0f, 0.0f) && near(y[3], -0.03f, 1e-7f));

    nk_sigmoid(x, y, 4);
    CHECK(near(y[0], 0.2689414f, 1e-5f));
    CHECK(near(y[1], 0.5f, 1e-6f));
    CHECK(near(y[2], 0.8807971f, 1e-5f));

    nk_tanh(x, y, 4);
    CHECK(near(y[0], tanhf(-1.0f), 1e-6f));
    CHECK(near(y[1], 0.0f, 1e-6f));

    nk_gelu(x, y, 4);
    CHECK(near(y[1], 0.0f, 1e-6f));       /* gelu(0) == 0 */
    CHECK(near(y[2], 1.9545f, 1e-3f));    /* gelu(2) = 1.9544997... */

    nk_gelu_tanh(x, y, 4);
    CHECK(near(y[2], 1.95457f, 1e-4f));
}

static void test_softmax(void)
{
    float s[3] = { 1.0f, 2.0f, 3.0f };
    nk_softmax(s, s, 3); /* in-place aliasing must work */
    CHECK(near(s[0] + s[1] + s[2], 1.0f, 1e-5f));
    CHECK(s[2] > s[1] && s[1] > s[0]);

    /* Large offsets must not overflow. */
    float big[3] = { 1000.0f, 1001.0f, 1002.0f };
    nk_softmax_inplace(big, 3);
    CHECK(!isnan(big[0]) && !isnan(big[1]) && !isnan(big[2]));
    CHECK(near(big[0] + big[1] + big[2], 1.0f, 1e-5f));
    CHECK(near(big[2], 0.6652410f, 1e-5f));

    float m[6] = { 0, 1, 0, 0, 1, 0 };
    nk_softmax_rows(m, 2, 3);
    CHECK(near(m[0] + m[1] + m[2], 1.0f, 1e-5f));
    CHECK(near(m[3] + m[4] + m[5], 1.0f, 1e-5f));
}

int main(void)
{
    test_matmul();
    test_activations();
    test_softmax();
    if (failures) {
        printf("%d test(s) failed\n", failures);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
