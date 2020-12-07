#pragma once

template <typename T>
void diff(T *out, const T *a, const T *b, int n) {
        for (int x = 0; x < n; ++x) {
                T ax = a[x];
                T bx = b[x];
                T diff = ax > bx ? ax - bx : bx - ax;
                out[x] = diff;
        }
}
