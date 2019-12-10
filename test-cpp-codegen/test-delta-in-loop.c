int delta(int a, int b) { return a == b ? 1 : 0; }

// check that this generates an O(n^2 loop)
void foo(int M, int N, int* __restrict__ A, int* __restrict__ x, int *out) {
    // x = N x 1
    // A = M x N
    // y = (M x N) x (N x 1) = M x 1
    // y = Ax
    // y[a] = (>< b A[a, b] x[b])
    // dAx[i, j]/dy[k] = δk_i A[i, j] x[j]
    for(int i = 0; i < M; ++i) {
        for(int j = 0; j < N; ++j) {
            for(int k = 0; k < M; ++k) {
                // final array
                out[i + M * j + M * N * k] = delta(i, j) * A[i + M * j] * x[j];
            }
        }
    }
}
