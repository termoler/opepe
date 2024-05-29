#include <math.h>
#include <mpi.h>
#include <pthread.h>
#include <vector>

#define LISTS_COUNT 10
#define TASK_COUNT 1500
#define SOLVER_FINISHED_WORK (-1)
#define NO_TASKS_TO_SHARE (-2)
#define SENDING_TASKS 1111
#define SENDING_TASK_COUNT 2222
#define REQUEST 3333

struct threadArgs{
    int rank{};
    int size{};
    int remainingTasks{};
    pthread_mutex_t mutex{};
    pthread_t threads[2]{};
    std::vector<int> tasks;
};

void initTasks(std::vector<int>& tasks, int taskCount, int iteration, int rank, int size) {
    for(int i = 0; i < taskCount; i++) {
        tasks[i] = abs(rank - (iteration % size)) * 1000 * (30 - i % 30);
    }
}

void doTask(threadArgs* tArgs) {
    for (int i = 0; i < tArgs->remainingTasks; i++) {
        pthread_mutex_lock(&tArgs->mutex);
        int weight = tArgs->tasks[i];
        pthread_mutex_unlock(&tArgs->mutex);
        double res = 0;
        for (int j = 0; j < weight; j++) {
            res += sqrt(j);
        }
    }
    tArgs->remainingTasks = 0;
}

void* solver(void* args) {
    auto* tArgs = (threadArgs *)args;
    int rank = tArgs->rank;
    int size = tArgs->size;
    tArgs->tasks.resize(TASK_COUNT);
    double startTime, finishTime, iterationDuration;
    for (int i = 0; i < LISTS_COUNT; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        tArgs->remainingTasks = TASK_COUNT;
        startTime = MPI_Wtime();
        initTasks(tArgs->tasks, TASK_COUNT, i, rank, size);
        doTask(tArgs);
        int response;
        for (int procIdx = 0; procIdx < size; procIdx++) {
            if (procIdx != rank) {
                MPI_Send(&rank, 1, MPI_INT,
                         procIdx, REQUEST, MPI_COMM_WORLD);
                MPI_Recv(&response, 1, MPI_INT,
                         procIdx, SENDING_TASK_COUNT,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (response != NO_TASKS_TO_SHARE) {
                    MPI_Recv(&tArgs->tasks[0], response, MPI_INT,
                             procIdx, SENDING_TASKS,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    pthread_mutex_lock(&tArgs->mutex);
                    tArgs->remainingTasks = response;
                    pthread_mutex_unlock(&tArgs->mutex);
                    doTask(tArgs);
                }
            }
        }
        finishTime = MPI_Wtime();
        iterationDuration = finishTime - startTime;
        printf("iteration in time %f\n", iterationDuration);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    int signal = SOLVER_FINISHED_WORK;
    if (size != 1) {
        MPI_Send(&signal, 1, MPI_INT, rank, REQUEST, MPI_COMM_WORLD);
    }
    return nullptr;
}

void* reciever(void* args) {
    auto* tArgs = (threadArgs *)args;
    int rank = tArgs->rank;
    int rankSentRequest, answer, request;
    MPI_Status status;
    while (true) {
        MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST, MPI_COMM_WORLD, &status);
        if (request == SOLVER_FINISHED_WORK) {
            return nullptr;
        }
        rankSentRequest = request;
        pthread_mutex_lock(&tArgs->mutex);
        if (tArgs->remainingTasks >= 2) {
            answer = tArgs->remainingTasks / (rank + 1);
            tArgs->remainingTasks = tArgs->remainingTasks / (rank + 1);
            printf("sharing %d from %d to %d\n", answer, rank, rankSentRequest);
            MPI_Send(&answer, 1, MPI_INT, rankSentRequest, SENDING_TASK_COUNT,
                     MPI_COMM_WORLD);
            MPI_Send(&tArgs->tasks[TASK_COUNT - answer], answer, MPI_INT,
                     rankSentRequest, SENDING_TASKS, MPI_COMM_WORLD);
        } else {
            answer = NO_TASKS_TO_SHARE;
            MPI_Send(&answer, 1, MPI_INT, rankSentRequest, SENDING_TASK_COUNT, MPI_COMM_WORLD);
        }
        pthread_mutex_unlock(&tArgs->mutex);
    }
}

int main() {
    int thread;
    threadArgs args;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &thread);
    MPI_Comm_rank(MPI_COMM_WORLD, &args.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &args.size);
    pthread_mutex_init(&args.mutex, nullptr);
    double start = MPI_Wtime();
    pthread_create(&args.threads[0], nullptr, solver, &args);
    if (args.size != 1) {
        pthread_create(&args.threads[1], nullptr, reciever, &args);
        pthread_join(args.threads[1], nullptr);
    }
    pthread_join(args.threads[0], nullptr);
    double finish = MPI_Wtime() - start;
    double time;
    MPI_Reduce(&finish, &time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (args.rank == 0) {
        printf("---------------\nTime %f\n", time);
    }
    MPI_Finalize();
}
