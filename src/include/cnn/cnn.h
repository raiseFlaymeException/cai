#ifndef CNN_H
#define CNN_H
#if defined(CNN_IMPL) && !defined(CMAT_IMPL) && !defined(CNN_NO_AUTO_CMAT_IMPORT)
#define CMAT_IMPL
#endif
#include "cmat/cmat.h"

#ifndef CNN_ASSERT
#include <assert.h>
#define CNN_ASSERT(cond, msg) CMAT_ASSERT(cond, msg)
#endif // CNN_ASSERT

#ifndef CNN_MALLOC
#include <stdlib.h>
#define CNN_MALLOC(num_elem, size_elem) CMAT_MALLOC(num_elem, size_elem)
#endif // CNN_MALLOC

#ifndef CNN_FREE
#include <stdlib.h>
#define CNN_FREE(ptr) CMAT_FREE(ptr)
#endif // CNN_FREE

#include <math.h>

typedef struct {
    CMat  *ws;
    CMat  *bs;
    CMat  *as;
    size_t n;
} CNN;

typedef struct CNNOpt {
    CMatType (*activ)(CMatType);
    CMatType (*cost)(CNN *, const struct CNNOpt *, const CMat *, const CMat *);
    void (*gradient)(CNN *, CNN *, const struct CNNOpt *, const CMat *, const CMat *);
    size_t print_every;
    void  *userdata;
} CNNOpt;

// initialization:
#define CNN_init(cnn, ...)                                                                         \
    {                                                                                              \
        size_t layers_size[] = {__VA_ARGS__};                                                      \
        CNN_init_arr(cnn, layers_size, sizeof(layers_size) / sizeof(*layers_size));                \
    }
void CNN_init_arr(CNN *cnn, size_t *layers_size, size_t layer_size);
void CNN_deinit(CNN *cnn);

// basics:
void CNN_forward(CNN *cnn, const CNNOpt *opt, const CMat *in, CMat *out);
void CNN_learn(CNN *cnn, const CNNOpt *opt, const CMat *tin, const CMat *tout, size_t iter,
               CMatType rate);

// activ:
CMatType cnn_sigmoid(CMatType x);

// cost:
#define CNN_cost(cnn, opt, tin, tout) ((opt)->cost((cnn), (opt), (tin), (tout)))
CMatType CNN_cost_MSE(CNN *cnn, const CNNOpt *opt, const CMat *tin, const CMat *tout);

// gradient:
void CNN_gradient_descent_SIG_MSE(CNN *cnn, CNN *g, const struct CNNOpt *opt, const CMat *tin,
                                  const CMat *tout);
void CNN_finite_diff(CNN *cnn, CNN *g, const struct CNNOpt *opt, const CMat *tin, const CMat *tout);

// other:
void CNN_debug_print(CNN *cnn);

#endif // CNN_H
// #define CNN_IMPL
#ifdef CNN_IMPL

// initialization:
void CNN_init_arr(CNN *cnn, size_t *layers_size, size_t layer_size) {
    cnn->n  = layer_size - 1;
    cnn->ws = CNN_MALLOC(cnn->n, sizeof(*cnn->ws));
    CNN_ASSERT(cnn->ws, "malloc failed");
    cnn->bs = CNN_MALLOC(cnn->n, sizeof(*cnn->bs));
    CNN_ASSERT(cnn->bs, "malloc failed");
    cnn->as = CNN_MALLOC(cnn->n + 1, sizeof(*cnn->as));
    CNN_ASSERT(cnn->as, "malloc failed");

    CMat_init(cnn->as, 1, *layers_size);
    for (size_t i = 0; i < cnn->n; ++i) {
        CMat_init(cnn->as + i + 1, 1, layers_size[i + 1]);

        size_t m = cnn->as[i].ncol;
        size_t p = cnn->as[i + 1].ncol;

        CMat_init(cnn->ws + i, m, p);
        CMat_iterate(cnn->ws + i, row, col, val, *val = rand() / (double)RAND_MAX;);

        CMat_init(cnn->bs + i, 1, p);
        CMat_iterate(cnn->bs + i, row, col, val, *val = rand() / (double)RAND_MAX;);
    }
}
void CNN_deinit(CNN *cnn) {
    for (size_t i = 0; i < cnn->n; ++i) {
        CMat_deinit(cnn->as + i);
        CMat_deinit(cnn->ws + i);
        CMat_deinit(cnn->bs + i);
    }
    CMat_deinit(cnn->as + cnn->n);

    CNN_FREE(cnn->as);
    CNN_FREE(cnn->ws);
    CNN_FREE(cnn->bs);
}

// basics:
void CNN_forward(CNN *cnn, const CNNOpt *opt, const CMat *in, CMat *out) {
    if (in) { CMat_iterate2(cnn->as, in, row, col, val1, val2, *val1 = *val2;); }
    for (size_t i = 0; i < cnn->n; ++i) {
        CMat_dot(cnn->as + i + 1, cnn->as + i, cnn->ws + i);
        CMat_iterate2(cnn->as + i + 1, cnn->bs + i, row, col, val1, val2,
                      *val1 = opt->activ(*val1 + *val2););
    }
    if (out) { CMat_iterate2(out, cnn->as + cnn->n, row, col, val1, val2, *val1 = *val2;); }
}
void CNN_learn(CNN *cnn, const CNNOpt *opt, const CMat *tin, const CMat *tout, size_t iter,
               CMatType rate) {
    CNN     g;
    size_t  layer_size  = cnn->n + 1;
    size_t *layers_size = alloca(layer_size * sizeof(*layers_size));
    for (size_t i = 0; i < layer_size; ++i) { layers_size[i] = cnn->as[i].ncol; }
    CNN_init_arr(&g, layers_size, layer_size);

    for (size_t i = 0; i < iter; ++i) {
        opt->gradient(cnn, &g, opt, tin, tout);
        for (size_t j = 0; j < cnn->n; ++j) {
            CMat_iterate2(cnn->ws + j, g.ws + j, row, col, val1, val2, *val1 -= *val2 * rate;);
            CMat_iterate2(cnn->bs + j, g.bs + j, row, col, val1, val2, *val1 -= *val2 * rate;);
        }
        if (opt->print_every && !(i % opt->print_every)) {
            printf("[INFO]: iter: %d, cost: %lf\n", i, CNN_cost(cnn, opt, tin, tout));
        }
    }

    CNN_deinit(&g);
}

// activ:
CMatType cnn_sigmoid(CMatType x) { return 1 / (1 + exp(-x)); }

// cost:
CMatType CNN_cost_MSE(CNN *cnn, const CNNOpt *opt, const CMat *tin, const CMat *tout) {
    const CMat *cnn_out = cnn->as + cnn->n;
    CNN_ASSERT(tin->ncol == cnn->as->ncol, "ncol don't match");
    CNN_ASSERT(tout->ncol == cnn_out->ncol, "ncol don't match");
    CNN_ASSERT(tin->nrow == tout->nrow, "nrow don't match");

    CMat out;
    CMat_init(&out, 1, cnn_out->ncol);

    CMatType sum = 0;
    for (size_t i = 0; i < tin->nrow; ++i) {
        const CMat in = CMat_from_submat(tin, i, 0, 1, tin->ncol);
        CNN_forward(cnn, opt, &in, &out);
        for (size_t j = 0; j < tout->ncol; ++j) {
            CMatType d = CMat_at(&out, 0, j) - CMat_at(tout, i, j);
            sum += d * d;
        }
    }

    CMat_deinit(&out);
    return sum / tin->nrow;
}

// gradient:
void CNN_gradient_descent_SIG_MSE(CNN *cnn, CNN *g, const struct CNNOpt *opt, const CMat *tin,
                                  const CMat *tout) {
    // stollen from tsoding video:
    // https://www.youtube.com/watch?v=o7da9anmnMs&list=PLpM-Dvs8t0VZPZKggcql-MmjaBdZKeDMw

    CNN_ASSERT(tin->nrow == tout->nrow, "nrow don't match");
    CNN_ASSERT(tout->ncol == cnn->as[cnn->n].ncol, "ncol don't match");

    for (size_t i = 0; i < g->n; ++i) {
        CMat_iterate(g->ws + i, row, col, val, *val = 0;);
        CMat_iterate(g->bs + i, row, col, val, *val = 0;);
        CMat_iterate(g->as + i, row, col, val, *val = 0;);
    }
    CMat_iterate(g->as + g->n, row, col, val, *val = 0;);

    size_t n = tin->nrow;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < cnn->as->ncol; ++j) { CMat_at(cnn->as, 0, j) = CMat_at(tin, i, j); }
        CNN_forward(cnn, opt, NULL, NULL);

        for (size_t j = 0; j <= cnn->n; ++j) { CMat_iterate(g->as + j, row, col, val, *val = 0;); }

        for (size_t j = 0; j < tout->ncol; ++j) {
            CMat_at(g->as + g->n, 0, j) = CMat_at(cnn->as + cnn->n, 0, j) - CMat_at(tout, i, j);
        }

        for (size_t l = cnn->n; l > 0; --l) {
            for (size_t j = 0; j < cnn->as[l].ncol; ++j) {
                CMatType a  = CMat_at(cnn->as + l, 0, j);
                CMatType da = CMat_at(g->as + l, 0, j);

                CMat_at(g->bs + l - 1, 0, j) += 2 * da * a * (1 - a);
                for (size_t k = 0; k < cnn->as[l - 1].ncol; ++k) {
                    CMatType pa = CMat_at(cnn->as + l - 1, 0, k);
                    CMatType w  = CMat_at(cnn->ws + l - 1, k, j);

                    CMat_at(g->ws + l - 1, k, j) += 2 * da * a * (1 - a) * pa;
                    CMat_at(g->as + l - 1, 0, k) += 2 * da * a * (1 - a) * w;
                }
            }
        }
    }

    for (size_t i = 0; i < g->n; ++i) {
        CMat_iterate(g->ws + i, row, col, val, *val /= n;);
        CMat_iterate(g->bs + i, row, col, val, *val /= n;);
    }
}
void CNN_finite_diff(CNN *cnn, CNN *g, const struct CNNOpt *opt, const CMat *tin,
                     const CMat *tout) {
    CMatType eps = *(CMatType *)opt->userdata;
    CMatType c   = CNN_cost(cnn, opt, tin, tout);

    for (size_t i = 0; i < cnn->n; ++i) {
        CMat_iterate2(cnn->ws + i, g->ws + i, row, col, val1, val2, {
            CMatType old_w = *val1;
            *val1 += eps;

            *val2 = (CNN_cost(cnn, opt, tin, tout) - c) / eps;

            *val1 = old_w;
        });
        CMat_iterate2(cnn->bs + i, g->bs + i, row, col, val1, val2, {
            CMatType old_b = *val1;
            *val1 += eps;

            *val2 = (CNN_cost(cnn, opt, tin, tout) - c) / eps;

            *val1 = old_b;
        });
    }
}

// other:
void CNN_debug_print(CNN *cnn) {
    for (size_t i = 0; i < cnn->n; ++i) {
        printf("cnn.as[%d]: %dx%d\n", i, cnn->as[i].nrow, cnn->as[i].ncol);
        printf("cnn.ws[%d]: %dx%d\n", i, cnn->ws[i].nrow, cnn->ws[i].ncol);
        CMat_print(cnn->ws + i);
        printf("cnn.bs[%d]: %dx%d\n", i, cnn->bs[i].nrow, cnn->bs[i].ncol);
        CMat_print(cnn->bs + i);
    }
    printf("cnn.as[%d]: %dx%d\n", cnn->n, cnn->as[cnn->n].nrow, cnn->as[cnn->n].ncol);
    printf("cnn.n = %d\n", cnn->n);
}

#endif // CNN_IMPL
