#include <iostream>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdio.h>
#include <unistd.h>

using namespace std;

int main()
{
    // ftok to generate unique key
    key_t key = ftok("shmfile", 65);

    // shmget returns an identifier in shmid
    int shmid = shmget(key, 1024, 0666 | IPC_CREAT);

    // shmat to attach to shared memory
    char *str = (char *)shmat(shmid, (void *)0, 0);

    std::cout << "key: " << key << ", shmid: " << shmid << std::endl;
    for (size_t i = 0; i < 10; i++)
    {
        cout << "Write Data : ";
        gets(str);
        printf("Data written in memory: %s\n", str);
        sleep(10);
    }

    //detach from shared memory
    shmdt(str);

    return 0;
}
