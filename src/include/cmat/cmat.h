#ifndef CMAT_H
#define CMAT_H

#include <inttypes.h>
#include <stdbool.h>

// define CMAT_NO_PRINT to disable importing stdio at the cost of not having printing
#ifndef CMAT_NO_PRINT
#include <stdio.h>
#endif // CMAT_PRINT

///
/// @brief the type of a matrix
///
///
typedef double CMatType;
///
/// @brief a matrix
///
///
typedef struct {
    CMatType *data; /// @memberof data content of the matrix
    size_t    nrow; /// @memberof nrow the number of row
    size_t    ncol; /// @memberof ncol the number of col
    size_t stride;  /// @memberof stride the stride is how much we need to jump to get the next row
} CMat;

// define CMAT_ASSERT before including cmat to replace assert
#ifndef CMAT_ASSERT
#include <assert.h>
///
/// @brief like standard assert
///
/// @param cond if true fail with message else nothing
/// @param msg the message we print on error
///
#define CMAT_ASSERT(cond, msg) assert((cond) && (msg))
#endif // CMAT_ASSERT

// define CMAT_MALLOC before including cmat to replace malloc
#ifndef CMAT_MALLOC
#include <stdlib.h>
///
/// @brief like standard malloc
///
/// @param num_elem the number of element
/// @param size_elem the size of one element
///
#define CMAT_MALLOC(num_elem, size_elem) malloc((num_elem) * (size_elem))
#endif // CMAT_MALLOC

// define CMAT_FREE before including cmat to replace free
#ifndef CMAT_FREE
#include <stdlib.h>
///
/// @brief like standard free
///
/// @param ptr the pointer we need to free
///
#define CMAT_FREE(ptr) free(ptr)
#endif // CMAT_FREE

///
/// @brief initiliaze a matrix (O(1) ?) (allocate)
///
/// @param cmat an non initialize matrix we went to initiliaze
/// @param nrow the number of row
/// @param ncol the number of col
///
void CMat_init(CMat *cmat, size_t nrow, size_t ncol);
///
/// @brief deinitialize a matrix (O(1) ?) (free)
///
/// @param cmat matrix to deinitialize
///
#define CMat_deinit(cmat) (CMAT_FREE((cmat)->data))

///
/// @brief equivalent to accessing a pointer by adding the index (zero based) (O(1))
///
/// warning:
/// no bound check
///
/// example:
/// *CMat_pat(cmat, 1, 1) = 5;
///
/// @param cmat the matrix to access
/// @param row the row we want to access
/// @param col the col we want to access
///
#define CMat_pat(cmat, row, col) ((cmat)->data + (row) * (cmat)->stride + (col))
///
/// @brief equivalent to accessing a pointer by adding the index than dereferencing (zero based)
/// (O(1))
///
/// warning:
/// no bound check
///
/// example:
/// CMat_at(cmat, 1, 1) = 5;
///
/// @param cmat the matrix to access
/// @param row the row we want to access
/// @param col the col we want to access
///
#define CMat_at(cmat, row, col) (*CMat_pat(cmat, row, col))

///
/// @brief iterate over a matrix (O(n*m))
///
/// example:
/// CMat_iterate(cmat, row, col, val, printf("cmat[%d][%d] = %lf\n", row, col, *val););
///
/// @param cmat the matrix to iterate
/// @param row the name we went the row index to have
/// @param col the name we went the col index to have
/// @param val the name we went the ptr to a value at row, col to have
/// @param block the block to execute on every iteration
///
#define CMat_iterate(cmat, row, col, val, block)                                                   \
    for (size_t row = 0; row < (cmat)->nrow; ++row) {                                              \
        for (size_t col = 0; col < (cmat)->ncol; ++col) {                                          \
            CMatType *val = CMat_pat(cmat, row, col);                                              \
            block                                                                                  \
        }                                                                                          \
    }
///
/// @brief iterate over 2 matrix (O(n*m))
///
/// example:
/// CMat_iterate2(cmat1, cmat2, row, col, val1, val2, *val1 += *val2;);
///
/// requirement:
/// cmat1->nrow == cmat2->nrow && cmat1->ncol == cmat2->ncol
///
/// @param cmat1 the matrix 1 to iterate
/// @param cmat2 the matrix 2 to iterate
/// @param row the name we went the row index to have
/// @param col the name we went the col index to have
/// @param val1 the name we went the ptr to a value 1 at row, col to have
/// @param val2 the name we went the ptr to a value 2 at row, col to have
/// @param block the block to execute on every iteration
///
#define CMat_iterate2(cmat1, cmat2, row, col, val1, val2, block)                                   \
    CMAT_ASSERT((cmat1)->nrow == (cmat2)->nrow, "nrow don't match");                               \
    CMAT_ASSERT((cmat1)->ncol == (cmat2)->ncol, "ncol don't match");                               \
    for (size_t row = 0; row < (cmat1)->nrow; ++row) {                                             \
        for (size_t col = 0; col < (cmat1)->ncol; ++col) {                                         \
            CMatType *val1 = CMat_pat(cmat1, row, col);                                            \
            CMatType *val2 = CMat_pat(cmat2, row, col);                                            \
            block                                                                                  \
        }                                                                                          \
    }
///
/// @brief iterate over 3 matrix (O(n*m))
///
/// example:
/// CMat_iterate3(cmat1, cmat2, cmat3, row, col, val1, val2, val3, *val3 = *val1 + *val2;);
///
/// requirement:
/// cmat1->nrow == cmat2->nrow && cmat1->ncol == cmat2->ncol && cmat1->nrow == cmat3->nrow &&
/// cmat1->ncol == cmat3->ncol
///
/// @param cmat1 the matrix 1 to iterate
/// @param cmat2 the matrix 2 to iterate
/// @param cmat3 the matrix 3 to iterate
/// @param row the name we went the row index to have
/// @param col the name we went the col index to have
/// @param val1 the name we went the ptr to a value 1 at row, col to have
/// @param val2 the name we went the ptr to a value 2 at row, col to have
/// @param val3 the name we went the ptr to a value 3 at row, col to have
/// @param block the block to execute on every iteration
///
#define CMat_iterate3(cmat1, cmat2, cmat3, row, col, val1, val2, val3, block)                      \
    CMAT_ASSERT((cmat1)->nrow == (cmat2)->nrow, "nrow don't match");                               \
    CMAT_ASSERT((cmat1)->ncol == (cmat2)->ncol, "ncol don't match");                               \
    CMAT_ASSERT((cmat1)->nrow == (cmat3)->nrow, "nrow don't match");                               \
    CMAT_ASSERT((cmat1)->ncol == (cmat3)->ncol, "ncol don't match");                               \
    for (size_t row = 0; row < (cmat1)->nrow; ++row) {                                             \
        for (size_t col = 0; col < (cmat1)->ncol; ++col) {                                         \
            CMatType *val1 = CMat_pat(cmat1, row, col);                                            \
            CMatType *val2 = CMat_pat(cmat2, row, col);                                            \
            CMatType *val3 = CMat_pat(cmat3, row, col);                                            \
            block                                                                                  \
        }                                                                                          \
    }

///
/// @brief create a matrix from an array without heap allocating (O(1))
///
/// warning:
/// no bound check
///
/// example:
/// CMatType arr[] = {1, 2, 3, 4, 5, 6};
/// CMat cmat = CMat_from_arr(arr, 2, 3);
///
/// @param arr the array to get the matrix from
/// @param _nrow the number of row
/// @param _ncol the number of column
/// @return a stack allocated matrix
///
#define CMat_from_arr(arr, _nrow, _ncol)                                                           \
    ((CMat){.data = (CMatType *)arr, .ncol = _ncol, .nrow = _nrow, .stride = _ncol})

///
/// @brief create a matrix from a 2d array without heap allocating (O(1))
///
/// example:
/// CMatType arr[2][3] = {{1, 2, 3}, {4, 5, 6}};
/// CMat cmat = CMat_from_2darr(arr);
///
/// @param arr the 2d array to get the matrix from
/// @return a stack allocated matrix
///
#define CMat_from_2darr(arr)                                                                       \
    CMat_from_arr(arr, sizeof(*arr) ? sizeof(arr) / sizeof(*arr) : 0,                              \
                  sizeof(*arr) ? sizeof(*arr) / sizeof(**arr) : 0)
///
/// @brief create a matrix from part of an array (O(1))
///
/// warning:
/// -DON'T do CMat_from_subarr(cmat->data, 0, 1, 2, 2, 3);
///  IF cmat have a stride this is UB
///  DO CMat_from_submat(cmat, 0, 1, 2, 2)
/// -no bound check
///
/// example:
/// CMatType arr[2][3] = {{1, 2, 3}, {4, 5, 6}};
/// CMat cmat = CMat_from_subarr(arr, 0, 1, 2, 2, 3);
/// CMat_print(&cmat);
/// output:
/// --   --
/// | 2 3 |
/// | 5 6 |
/// --   --
///
/// @param arr the array to get the matrix from
/// @param row_start the starting point of the row
/// @param col_start the starting point of the column
/// @param _ncol the number of row we want
/// @param _ncol the number of col we want
/// @param _stride the number of colomn of the actual array
/// @return a stack allocated matrix
///
#define CMat_from_subarr(arr, row_start, col_start, _nrow, _ncol, _stride)                         \
    ((CMat){.data   = (CMatType *)(arr) + (row_start) * ((col_start) + (_ncol)) + (col_start),     \
            .nrow   = (_nrow),                                                                     \
            .ncol   = (_ncol),                                                                     \
            .stride = (_stride)})
///
/// @brief create a matrix from part of a 2d array (O(1))
///
/// warning:
/// -DON'T do CMat_from_sub2darr(cmat->data, 0, 1, 2, 2);
///  IF cmat have a stride this is UB
///  DO CMat_from_submat(cmat, 0, 1, 2, 2)
/// -no bound check
///
/// example:
/// CMatType arr[2][3] = {{1, 2, 3}, {4, 5, 6}};
/// CMat cmat = CMat_from_sub2darr(arr, 0, 1, 2, 2);
/// CMat_print(&cmat);
/// output:
/// --   --
/// | 2 3 |
/// | 5 6 |
/// --   --
///
/// @param arr the array to get the matrix from
/// @param row_start the starting point of the row
/// @param col_start the starting point of the column
/// @param _ncol the number of row we want
/// @param _ncol the number of col we want
/// @return a stack allocated matrix
///
#define CMat_from_sub2darr(arr, row_start, col_start, _nrow, _ncol)                                \
    ((CMat){.data   = (CMatType *)(arr) + (row_start) * ((col_start) + (_ncol)) + (col_start),     \
            .nrow   = (_nrow),                                                                     \
            .ncol   = (_ncol),                                                                     \
            .stride = (sizeof(*(arr)) / sizeof(**(arr)))})
///
/// @brief create a matrix from part of another matrix (O(1))
///
/// warning:
/// no bound check
///
/// example:
/// CMatType arr[2][3] = {{1, 2, 3}, {4, 5, 6}};
/// CMat cmat = CMat_from_subarr(arr, 0, 1, 2, 2);
/// CMat cmat2 = CMat_from_submat(arr, 1, 0, 1, 2);
/// CMat_print(&cmat2);
/// output:
/// --   --
/// | 5 6 |
/// --   --
///
/// @param cmt the matrix to get the matrix from
/// @param row_start the starting point of the row
/// @param col_start the starting point of the column
/// @param _ncol the number of row we want
/// @param _ncol the number of col we want
/// @return a stack allocated matrix
///
#define CMat_from_submat(cmat, row_start, col_start, _nrow, _ncol)                                 \
    ((CMat){.data = (CMatType *)((cmat)->data) + (row_start) * ((col_start) + (cmat)->stride) +    \
                    (col_start),                                                                   \
            .nrow   = (_nrow),                                                                     \
            .ncol   = (_ncol),                                                                     \
            .stride = (col_start + (cmat)->stride)})

///
/// @brief populate an identity matrix into dst (O(n^2))
///
/// example:
/// CMat identity;
/// CMat_init(&identity, 3, 3);
/// CMat_identity(&identity);
/// CMat_print(&identity);
/// output:
/// --     --
/// | 1 0 0 |
/// | 0 1 0 |
/// | 0 0 1 |
/// --     --
///
/// requirement:
/// dst->nrow == dst->ncol
///
/// @param dst the matrix destination provide the size, should have enought space to store the
/// resulted matrix and should be a square matrix
///
void CMat_identity(CMat *dst);

///
/// @brief duplicate a matrix that need to be deinit with CMat_deinit (O(n*m)) (allocate)
///
/// @param src the matrix to duplicate
/// @return the duplicated matrix
///
#define CMat_dup(src)                                                                              \
    ({                                                                                             \
        CMat dst;                                                                                  \
        CMat_init(&dst, (src)->nrow, (src)->ncol);                                                 \
        CMat_iterate2(&dst, (src), row, col, dst_val, src_val, *dst_val = *src_val;);              \
        dst;                                                                                       \
    })

///
/// @brief transpose a matrix (O(n*m))
///
/// example:
/// CMatType arr[2][3] = {{1, 2, 3}, {4, 5, 6}};
/// CMat cmat = CMat_from_2darr(arr);
/// CMatType arr_transpose[3][2];
/// CMat cmat_transpose = CMat_from_2darr(arr_transpose);
/// CMat_transpose(&cmat_transpose, &cmat);
/// CMat_print(&cmat_transpose);
/// output:
/// --   --
/// | 1 2 |
/// | 3 4 |
/// | 5 6 |
/// --   --
///
/// requirement:
/// dst->nrow == src->ncol && dst->ncol == src->nrow
///
/// @param dst the resulted matrix
/// @param src the matrix to transpose
///
#define CMat_transpose(dst, src)                                                                   \
    CMAT_ASSERT((dst)->nrow == (src)->ncol, "dst->nrow should be == to src->ncol");                \
    CMAT_ASSERT((dst)->ncol == (src)->nrow, "dst->ncol should be == to src->nrow");                \
    CMat_iterate((src), row, col, src_val, CMat_at(dst, col, row) = *src_val;);

///
/// @brief do a dot product between 2 matrix and put the result into dst (O(n*m^2))
///
/// example:
/// CMatType arr1[2][3] = {{1, 2, 3}, {4, 5, 6}};
/// CMat     cmat1      = CMat_from_2darr(arr1);
///
/// CMatType arr2[3][2] = {{10, 11}, {20, 21}, {30, 31}};
/// CMat     cmat2      = CMat_from_2darr(arr2);
///
/// CMatType arr3[2][2];
/// CMat     cmat3 = CMat_from_2darr(arr3);
///
/// CMat_dot(&arr3, &arr1, &arr2);
/// CMat_print(&arr3);
/// output:
/// --       --
/// | 140 146 |
/// | 320 335 |
/// --       --
///
/// requirement:
/// cmat1->nrow == dst->nrow && cmat1->ncol == cmat2->nrow && cmat2->ncol == dst->ncol
///
/// @param dst the resulted matrix
/// @param cmat1 the first matrix
/// @param cmat2 the second matrix
///
void CMat_dot(CMat *dst, const CMat *cmat1, const CMat *cmat2);

///
/// @brief cofactor of point row, col in matrix cmat (O(n!)) (allocate and free)
///
/// requirement:
/// cmat->nrow == cmat->ncol
///
/// @param cmat the matrix we need to get the cofactor of
/// @param row the row of point of the matrix we need to get the cofactor of
/// @param col the col of point of the matrix we need to get the cofactor of
/// @return the cofactor
///
CMatType CMat_cofactor(const CMat *cmat, size_t row, size_t col);
///
/// @brief determinant of matrix (Laplace expansion) (O(n!))
///
/// example:
/// CMatType arr[4][4] = {{-1, 0, 0, -2}, {1, 0, 5, -5}, {0, 1, 4, 0}, {0, 0, -5, 0}};
/// CMat     cmat      = CMat_from_2darr(arr);
/// CMatType det      = CMat_det(&cmat);
/// printf("%lf\n", det);
/// output:
/// -35
///
/// requirement:
/// cmat->nrow == cmat->ncol
///
/// @param cmat the matrix we need to get the determinant of
/// @return the determinant
///
CMatType CMat_det(const CMat *cmat);
///
/// @brief adjugate of matrix (O(n!))
///
/// requirement:
/// src->nrow == src->ncol && dst->nrow == src->nrow && dst->ncol == src->ncol
///
/// @param dst the destination of the adjugate
/// @param src the matrix to get the adjugate of
///
void CMat_adj(CMat *dst, const CMat *src);

///
/// @brief inverse of matrix (Gaussian Jordan elemination) (O(n^3)) (allocate and free)
///
/// example:
/// CMatType arr[4][4] = {{-1, 0, 0, -2}, {1, 0, 5, -5}, {0, 1, 4, 0}, {0, 0, -5, 0}};
/// CMat     cmat      = CMat_from_2darr(arr);
///
/// if (CMat_inverse(&cmat)) {
///     CMat_print(&cmat);
/// } else {
///     printf("no inverse");
/// }
/// output:
/// --                    --
/// | -5/7   2/7   0   2/7 |
/// |    0     0   1   4/5 |
/// |    0     0   0  -1/5 |
/// | -1/7  -1/7   0  -1/7 |
/// --                    --
///
/// requirement:
/// det != 0 && dst->nrow == dst->ncol
///
/// @param cmat the matrix to get the inverse of and to get the result
/// @return true if no error else false
///
bool CMat_inverse(CMat *cmat);

#ifndef CMAT_NO_PRINT
///
/// @brief print a matrix to the file f using the precision float_pres (allocate and free)
///
/// @param f the file to print to
/// @param cmat the matrix to print
/// @param float_pres the precision of the floating point value printing
///
void CMat_fprint_pres(FILE *f, const CMat *cmat, size_t float_pres);
#endif // CMAT_NO_PRINT

// the default precision of printing of floating point value
#define CMAT_FPRINT_DEFAULT_PRES 6
///
/// @brief print a matrix to the file f using the precision CMAT_FPRINT_DEFAULT_PRES (allocate and
/// free)
///
/// @param f the file to print to
/// @param cmat the matrix to print
///
#define CMat_fprint(f, cmat) CMat_fprint_pres(f, cmat, CMAT_FPRINT_DEFAULT_PRES)
///
/// @brief print a matrix to stdout using the precision float_pres (allocate and free)
///
/// @param cmat the matrix to print
/// @param float_pres the precision of the floating point value printing
///
#define CMat_print_pres(cmat, pres) CMat_fprint_pres(stdout, cmat, pres)
///
/// @brief print a matrix to stdout using the precision CMAT_FPRINT_DEFAULT_PRES (allocate and free)
///
/// @param cmat the matrix to print
///
#define CMat_print(cmat) CMat_fprint(stdout, cmat)

#endif // CMAT_H
// #define CMAT_IMPL
#ifdef CMAT_IMPL

void CMat_init(CMat *cmat, size_t nrow, size_t ncol) {
    cmat->data = CMAT_MALLOC(nrow * ncol, sizeof(*cmat->data));
    CMAT_ASSERT(cmat->data, "malloc failed");

    cmat->nrow   = nrow;
    cmat->ncol   = ncol;
    cmat->stride = ncol;
}

void CMat_identity(CMat *dst) {
    CMAT_ASSERT(dst->nrow == dst->ncol, "row and col don't match");
    CMat_iterate(dst, row, col, val, {
        if (row == col) {
            *val = 1;
        } else {
            *val = 0;
        }
    });
}

void CMat_dot(CMat *dst, const CMat *cmat1, const CMat *cmat2) {
    CMAT_ASSERT(cmat1->nrow == dst->nrow, "a->nrow should match with dst->nrow");
    CMAT_ASSERT(cmat1->ncol == cmat2->nrow, "a->ncol should match with b->nrow");
    CMAT_ASSERT(cmat2->ncol == dst->ncol, "b->ncol should match with dst->ncol");

    for (size_t col = 0; col < dst->ncol; ++col) {
        for (size_t row = 0; row < dst->nrow; ++row) {
            CMatType sum = 0;
            for (size_t i = 0; i < cmat1->ncol; ++i) {
                sum += CMat_at(cmat1, row, i) * CMat_at(cmat2, i, col);
            }
            CMat_at(dst, row, col) = sum;
        }
    }
}

CMatType CMat_cofactor(const CMat *cmat, size_t row, size_t col) {
    CMat submat;
    CMat_init(&submat, cmat->nrow - 1, cmat->ncol - 1);

    size_t idx = 0;
    for (size_t i = 0; i < cmat->nrow; ++i) {
        for (size_t j = 0; j < cmat->ncol; ++j) {
            if (i == row || j == col) { continue; }
            // no stride so it's fine
            submat.data[idx++] = CMat_at(cmat, i, j);
        }
    }

    CMatType det = CMat_det(&submat);

    CMat_deinit(&submat);
    return ((row + col) % 2 == 0) ? det : -det;
}
CMatType CMat_det(const CMat *cmat) {
    CMAT_ASSERT(cmat->nrow == cmat->ncol, "the determinent is only defined for square matrices");
    // Laplace expansion

    // base case: 0x0
    if (cmat->nrow == 0) { return 1; }

    // recursive_case: nxn => (n-1)x(n-1)
    CMatType det = 0;
    for (size_t col = 0; col < cmat->ncol; ++col) {
        // CMat_cofactor call CMat_det
        det += CMat_at(cmat, 0, col) * CMat_cofactor(cmat, 0, col);
    }
    return det;
}
void CMat_adj(CMat *dst, const CMat *src) {
    CMAT_ASSERT(src->nrow == src->ncol, "adj only defined for square matrix");
    CMAT_ASSERT(dst->nrow == src->nrow, "nrow don't match");
    CMAT_ASSERT(dst->ncol == src->ncol, "ncol don't match");

    CMat_iterate(src, row, col, unused, {
        (void)unused;
        CMat_at(dst, col, row) = CMat_cofactor(src, row, col);
    });
}
bool CMat_inverse(CMat *cmat) {
    CMat id;
    CMat_init(&id, cmat->nrow, cmat->ncol);
    CMat_identity(&id);

    for (size_t i = 0; i < cmat->nrow; ++i) {
        if (CMat_at(cmat, i, i) == 0) {
            bool   found = false;
            size_t j     = i + 1;
            for (; j < cmat->nrow; ++j) {
                if (CMat_at(cmat, j, i) != 0) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                CMat_deinit(&id);
                return false;
            }

            for (size_t k = 0; k < cmat->ncol; ++k) {
                CMatType temp       = CMat_at(cmat, i, k);
                CMat_at(cmat, i, k) = CMat_at(cmat, j, k);
                CMat_at(cmat, j, k) = temp;

                temp               = CMat_at(&id, i, k);
                CMat_at(&id, i, k) = CMat_at(&id, j, k);
                CMat_at(&id, j, k) = temp;
            }
        }

        CMatType div = CMat_at(cmat, i, i);
        for (size_t col = 0; col < cmat->ncol; ++col) { CMat_at(cmat, i, col) /= div; }
        for (size_t col = 0; col < id.ncol; ++col) { CMat_at(&id, i, col) /= div; }

        for (size_t row = 0; row < cmat->nrow; ++row) {
            if (row == i) { continue; }
            CMatType x = CMat_at(cmat, row, i);
            for (size_t col = 0; col < cmat->ncol; ++col) {
                CMat_at(cmat, row, col) -= CMat_at(cmat, i, col) * x;
            }
            for (size_t col = 0; col < id.ncol; ++col) {
                CMat_at(&id, row, col) -= CMat_at(&id, i, col) * x;
            }
        }
    }

    CMat_iterate2(cmat, &id, row, col, val1, val2, *val1 = *val2;);

    CMat_deinit(&id);
    return true;
}

static size_t str_size_f(CMatType f, size_t float_pres) {
    // we don't print -0.0
    if (f == -0.0) { f = 0.0; }
    //                     precision  + . + correct error if -1 < f < 1 + -
    size_t cur_elem_size = float_pres + 1 + (-1 < f && f < 1) + (f < 0);
    for (long long n = f; n; n /= 10) { ++cur_elem_size; }
    return cur_elem_size;
}

#ifndef CMAT_NO_PRINT
void CMat_fprint_pres(FILE *f, const CMat *cmat, size_t float_pres) {
    size_t *max_elems_size = CMAT_MALLOC(cmat->ncol, sizeof(*max_elems_size));
    CMAT_ASSERT(max_elems_size, "malloc failed");

    size_t max_row_size = cmat->ncol - 1;
    for (size_t col = 0; col < cmat->ncol; ++col) { max_elems_size[col] = 0; }

    for (size_t row = 0; row < cmat->nrow; ++row) {
        for (size_t col = 0; col < cmat->ncol; ++col) {
            size_t cur_elem_size = str_size_f(CMat_at(cmat, row, col), float_pres);

            if (cur_elem_size > max_elems_size[col]) {
                max_row_size += cur_elem_size - max_elems_size[col];
                max_elems_size[col] = cur_elem_size;
            }
        }
    }

    fprintf(f, "--%*s--\n", max_row_size, "");

    for (size_t row = 0; row < cmat->nrow; ++row) {
        for (size_t col = 0; col < cmat->ncol; ++col) {
            if (col == 0) { fprintf(f, "| "); }

            CMatType cur_elem      = CMat_at(cmat, row, col);
            size_t   cur_elem_size = str_size_f(cur_elem, float_pres);

            fprintf(f, "%*s%.*lf ", max_elems_size[col] - cur_elem_size, "", float_pres, cur_elem);
        }
        fputs("|\n", f);
    }
    fprintf(f, "--%*s--\n", max_row_size, "");

    CMAT_FREE(max_elems_size);
}
#endif // CMAT_NO_PRINT
#endif // CMAT_IMPL
