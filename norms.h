#pragma once
#include <math.h>

template <typename T>
double l2norm(T *a, T *b, int n, int relative = 0) {
        double err=0;
        double norm_a = 0;
        for (int i = 0; i < n; i++) {
                err += (a[i] - b[i]) * (a[i] - b[i]);
                if (relative) norm_a += a[i] * a[i];
        }
        err = sqrt(err);
        if (relative) err /= sqrt(norm_a);
        return err;
}

template <typename T>
double l1norm(T *a, T *b, int n, int relative = 0) {
        double err=0;
        double norm_a = 0;
        for (int i = 0; i < n; i++) {
                err = a[i] - b[i] > 0 ? a[i] - b[i] : b[i] - a[i];
                if (relative) norm_a += a[i] > 0 ? a[i] : - a[i];
        }
        if (relative) err /= norm_a;
        return err;
}

template <typename T>
double linfnorm(T *a, T *b, int n, int relative = 0) {
        double err = 0;
        double norm_a = 0;
        for (int i = 0; i < n; i++) {
                double diff = a[i] - b[i] > 0 ? a[i] - b[i] : b[i] - a[i];
                err = err > diff ? err : diff;
                if (relative) { 
                        double diff_a = a[i] > 0 ? a[i] : - a[i];
                        norm_a = norm_a > diff_a ? norm_a : diff_a;
                }
        }
        if (relative) err /= norm_a;
        return err;
}
