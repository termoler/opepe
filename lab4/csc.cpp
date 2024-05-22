#include <cmath>
#include <mpi.h>
#include <vector>

#define N 100

#define X_0 -1.0
#define Y_0 -1.0
#define Z_0 -1.0

#define D_X 2.0
#define D_Y 2.0
#define D_Z 2.0

#define H_X (D_X / (N - 1))
#define H_Y (D_Y / (N - 1))
#define H_Z (D_Z / (N - 1))

#define H_X2 (H_X * H_X)
#define H_Y2 (H_Y * H_Y)
#define H_Z2 (H_Z * H_Z)

#define A 1.0E6
#define EPSILON (double)1.0E-5
#define DENOMINATOR (2 / H_X2 + 2 / H_Y2 + 2 / H_Z2 + A)

double functionF(double x, double y, double z) {
    return x * x + y * y + z * z;
}
double functionR(double x, double y, double z) {
    return 6 - A * functionF(x, y, z);
}
int IDX(int i, int j, int k) {
    return i * N * N + j * N + k;
}
double getX(int i) {
    return X_0 + i * H_X;
}
double getY(int j) {
    return Y_0 + j * H_Y;
}
double getZ(int k) {
    return Z_0 + k * H_Z;
}

void divideArea(std::vector<int>& layerHeights, std::vector<int>& displs, int size) {
    for (int i = 0; i < size; ++i) {
        layerHeights[i] = (N % size > i) ? (N / size + 1) : (N / size);
        displs[i] = (i >= 1) ? (displs[i - 1] + layerHeights[i - 1]) : 0;
    }
}

void initLayers(std::vector<double>& currentFunc, int layerHeight, int displ) {
    for (int i = 0; i < layerHeight; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                if ((displ + i == 0) || (j == 0) || (k == 0) || (displ + i == N - 1) || (j == N - 1) || (k == N - 1)) {
                    currentFunc[IDX(i, j, k)] = functionF(getX(displ + i), getY(j), getZ(k));
                }
            }
        }
    }
}

double calculationCenter(std::vector<double>& previousFunc, std::vector<double>& currentFunc, int layerHeight, int displ) {
    double maxDiff = 0.0;
    double tmpMaxDiff = 0.0;
    for (int i = 1; i < layerHeight - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            for (int k = 1; k < N - 1; ++k) {
                double partValue = (previousFunc[IDX(i + 1, j, k)] + previousFunc[IDX(i - 1, j, k)]) / H_X2
                                   + (previousFunc[IDX(i, j + 1, k)] + previousFunc[IDX(i, j - 1, k)]) / H_Y2
                                   + (previousFunc[IDX(i, j, k + 1)] + previousFunc[IDX(i, j, k - 1)]) / H_Z2;
                currentFunc[IDX(i, j, k)] = (partValue - functionR(getX(displ + i), getY(j), getZ(k))) / DENOMINATOR;
                tmpMaxDiff = fabs(currentFunc[IDX(i, j, k)] - previousFunc[IDX(i, j, k)]);
                maxDiff = fmax(maxDiff, tmpMaxDiff);
            }
        }
    }
    return maxDiff;
}

double calculationBorder(std::vector<double>& previousFunc, std::vector<double>& currentFunc, std::vector<double>& upBorder, std::vector<double>& downBorder, int layerHeight, int displ, int rank, int size) {
    double maxDiff = 0.0;
    double tmpMaxDiff = 0.0;
    for (int j = 1; j < N - 1; ++j)
        for (int k = 1; k < N - 1; ++k) {
            // Верхние границы
            if (rank != 0) {
                double partSum = (previousFunc[IDX(1, j, k)] + upBorder[IDX(0, j, k)]) / H_X2
                                 + (previousFunc[IDX(0, j + 1, k)] + previousFunc[IDX(0, j - 1, k)]) / H_Y2
                                 + (previousFunc[IDX(0, j, k + 1)] + previousFunc[IDX(0, j, k - 1)]) / H_Z2;
                currentFunc[IDX(0, j, k)] = (partSum - functionR(getX(displ), getY(j), getZ(k))) / DENOMINATOR;
                tmpMaxDiff = fabs(currentFunc[IDX(0, j, k)] - previousFunc[IDX(0, j, k)]);
            }
            // Нижние границы
            if (rank != size - 1) {
                double partSum = (previousFunc[IDX(layerHeight - 2, j, k)] + downBorder[IDX(0, j, k)]) / H_X2
                        + (previousFunc[IDX(layerHeight - 1, j + 1, k)] + previousFunc[IDX(layerHeight - 1, j - 1, k)]) / H_Y2
                        + (previousFunc[IDX(layerHeight - 1, j, k + 1)] + previousFunc[IDX(layerHeight - 1, j, k - 1)]) / H_Z2;
                currentFunc[IDX(layerHeight - 1, j, k)] = (partSum - functionR(getX(displ + layerHeight - 1), getY(j), getZ(k))) / DENOMINATOR;
                tmpMaxDiff = fabs(currentFunc[IDX(layerHeight - 1, j, k)] - previousFunc[IDX(layerHeight - 1, j, k)]);
            }
            maxDiff = fmax(tmpMaxDiff, maxDiff);
        }

    return maxDiff;
}

double calculationMaxDiff(const std::vector<double>& currentFunc, int layerHeight, int displ) {
    double tmpMaxDelta = 0.0;
    double maxDelta = 0.0;
    for (int i = 0; i < layerHeight; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                tmpMaxDelta = fabs(currentFunc[IDX(i, j, k)] - functionF(getX(displ + i), getY(j), getZ(k)));
                maxDelta = fmax(tmpMaxDelta, maxDelta);
            }
        }
    }
    return maxDelta;
}

int main(int argc, char **argv) {
    int rank = 0, size = 0;
    double startTime, finishTime = 0.0;
    double lastMaxDiff = EPSILON, maxDiff = 0.0;

    MPI_Request upReq[2];
    MPI_Request downReq[2];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<int> layerHeights(size);
    std::vector<int> displs(size, 0);

    divideArea(layerHeights, displs, size);

    std::vector<double> previousFunc(layerHeights[rank] * N * N, 0);
    std::vector<double> currentFunc(layerHeights[rank] * N * N, 0);

    initLayers(currentFunc, layerHeights[rank], displs[rank]);
    previousFunc = currentFunc;

    std::vector<double> currentUpBorder(N * N);
    std::vector<double> currentDownBorder(N * N);
    std::vector<double> previousUpBorder(N * N);
    std::vector<double> previousDownBorder(N * N);

    startTime = MPI_Wtime();

    while (true) {
        std::swap(previousFunc, currentFunc);
        if (rank != 0) {
            std::copy(previousFunc.begin(), previousFunc.begin() + N * N, previousUpBorder.begin());
            MPI_Isend(&previousUpBorder[0], N * N, MPI_DOUBLE, rank - 1, rank, MPI_COMM_WORLD, &upReq[0]);
            MPI_Irecv(&currentUpBorder[0], N * N, MPI_DOUBLE, rank - 1, rank - 1, MPI_COMM_WORLD, &upReq[1]);
        }
        if (rank != size - 1) {
            std::copy(previousFunc.begin() + (layerHeights[rank] - 1) * N * N, previousFunc.begin() + layerHeights[rank] * N * N, previousDownBorder.begin());
            MPI_Isend(&previousDownBorder[0], N * N, MPI_DOUBLE, rank + 1, rank, MPI_COMM_WORLD, &downReq[0]);
            MPI_Irecv(&currentDownBorder[0], N * N, MPI_DOUBLE, rank + 1, rank + 1, MPI_COMM_WORLD, &downReq[1]);
        }
        double tmpMaxDiffCenter = calculationCenter(previousFunc, currentFunc, layerHeights[rank], displs[rank]);
        if (rank != 0) {
            MPI_Waitall(2, upReq, MPI_STATUS_IGNORE);
        }
        if (rank != size - 1) {
            MPI_Waitall(2, downReq, MPI_STATUS_IGNORE);
        }

        double tmpMaxDiffBorder = calculationBorder(previousFunc, currentFunc, currentUpBorder, currentDownBorder, layerHeights[rank], displs[rank], rank, size);

        maxDiff = lastMaxDiff;
        lastMaxDiff = fmax(tmpMaxDiffCenter, tmpMaxDiffBorder);
        if(!(maxDiff >= EPSILON)) break;
    }
    double tmpMaxDiff = calculationMaxDiff(previousFunc, layerHeights[rank], displs[rank]);
    MPI_Allreduce(&tmpMaxDiff, &maxDiff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    finishTime = MPI_Wtime();
    if (rank == 0) {
        printf("Time: %lf\n", finishTime - startTime);
        printf("Max difference: %lf\n", maxDiff);
    }
    MPI_Finalize();
    return 0;
}
