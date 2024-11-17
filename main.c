#include <time.h>

/* TODO:
 *  [ ] bug where it randomly doesn't works
 *  [ ] CNN_print and delete CNN_debug_print
 *  [ ] documentation
 *  [ ] instead of allocating a lot of small matricies, allocating a big chunk of memory and
 * creating a lot of submatrices from this memory so memory access is faster and it's easier for
 * serialization
 *  [ ] serialization and deserialization
 *  [ ] test with a bigger neural network
 *  */

#define CNN_IMPL
#include "cnn/cnn.h"

int main() {
    srand(time(NULL));

    CMatType tarr[4][3] = {
        {0, 0, 0},
        {1, 0, 1},
        {0, 1, 1},
        {1, 1, 0},
    };

    CMat tin  = CMat_from_sub2darr(tarr, 0, 0, 4, 2);
    CMat tout = CMat_from_sub2darr(tarr, 2, 0, 4, 1);

    CNN cnn;
    CNN_init(&cnn, 2, 2, 1);

    CNNOpt opt = {.activ       = cnn_sigmoid,
                  .cost        = CNN_cost_MSE,
                  .gradient    = CNN_gradient_descent_SIG_MSE,
                  .print_every = 1e2};

    CNN_learn(&cnn, &opt, &tin, &tout, 1e4, 1e-1);

    CNN_debug_print(&cnn);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            CMatType in_arr[1][2] = {{i, j}};
            CMat     in           = CMat_from_2darr(in_arr);

            CMatType out_arr[1][1];
            CMat     out = CMat_from_2darr(out_arr);

            CNN_forward(&cnn, &opt, &in, &out);

            printf("%d | %d = %lf\n", i, j, *out.data);
        }
    }

    CNN_deinit(&cnn);
    return 0;
}
