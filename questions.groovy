
for (int i = 0; i < round(N/T); i++) {
    for (int j = 0; j < round(N/T); j++) {
        //each block computes a c tile
        Initialize C_tile to 0;
        for (int k = 0; k < round(N/T); k++) {
            //a tile and b tile are loaded
            __syncthreads();
            //compute partial matmul
            for (int t = 0; t < T; t++) {
                C_tile += matrixMultiply(A_tile, B_tile)
            }
            __syncthreads();
        }
        //write the c tile back to global memory
    }
}

for (int i = 0; i < round(N/B); i++) {
    for (int j = 0; j < round(N/B); j++) {
        //each block computes a C block
        C_block = zeros(B, B);
        for (int k = 0; k < round(N/B); k++) {
            //load A and B into global memory
            __syncthreads();
            //compute partial matmul
            A_block = loadBlock(A, i, k);
            B_block = loadBlock(B, k, j);
            C_block += matrixMultiply(A_block, B_block); //perform kernel operations on A and B blocks and store to C
            }
            __syncthreads();
        }
        writeBlock(C, i, j, C_block); //write to device or host memory
    }
