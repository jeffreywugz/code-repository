#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/fcntl.h>
#include <semaphore.h>
#include <pthread.h>
#include "device_info_collector.h"
#include "util.h"

static char* get_request_sem_file_name(char* file_name, int id)
{
        return str_template(file_name, "request-%x.sem", id);
}

static char* get_finish_sem_file_name(char* file_name, int id)
{
        return str_template(file_name, "finish-%x.sem", id);
}

static int _device_info_collector_collect(
        DeviceInfoCollector* device_info_collector)
{
        CUresult r;
        unsigned int free_mem, total_mem;
        long long time_start, time_end;
        long long delay;

        time_start = get_time();
        r = cuMemGetInfo(&free_mem, &total_mem);
        time_end = get_time();
        delay = time_end - time_start;
        
        if(r != CUDA_SUCCESS)panic("cuMemGetInfo\n");
        device_info_collector->device_info.free_mem = free_mem;
        device_info_collector->device_info.total_mem = total_mem;
        device_info_collector->device_info.delay = delay;
        return 0;
}

static int device_info_collector_handle(
        DeviceInfoCollector* device_info_collector)
{
        sem_wait(device_info_collector->request_sem);
        _device_info_collector_collect(device_info_collector);
        sem_post(device_info_collector->finish_sem);
        return 0;
}

static int device_info_collector_init_cuda_ctx(
        DeviceInfoCollector* device_info_collector)
{
        CUresult r;
        r = cuInit(0);
        if(r != CUDA_SUCCESS)panic("cuInit\n");
        r = cuCtxCreate(&device_info_collector->cu_ctx, 0,
                        device_info_collector->id);
        if(r != CUDA_SUCCESS)panic("cuCtxCreate\n");
        return 0;
}

static int device_info_collector_daemon_main_loop(
        DeviceInfoCollector* device_info_collector)
{
        device_info_collector_init_cuda_ctx(device_info_collector);
        while(1)
                device_info_collector_handle(device_info_collector);
        return 0;
}

static int device_info_collector_init_daemon(
        DeviceInfoCollector* device_info_collector)
{
        int r;
        pthread_t thread;
        r = pthread_create(&thread, NULL,
                           (void*)device_info_collector_daemon_main_loop,
                           (void*)device_info_collector);
        if(r < 0)sys_panic("pthread_create");
        return 0;
}

static int device_info_collector_init_request_sem(
        DeviceInfoCollector* device_info_collector)
{
        int request_value = 0;
        char request_sem_file_name[MAX_FILE_NAME_LEN];
        sem_t* sem;
        get_request_sem_file_name(request_sem_file_name,
                                  device_info_collector->id);
        sem = sem_open(request_sem_file_name, O_CREAT, S_IRWXU, request_value);
        if(sem == SEM_FAILED)sys_panic("sem_open");
        sem_unlink(request_sem_file_name);
        device_info_collector->request_sem = sem;
        return 0;
}

static int device_info_collector_init_finish_sem(
        DeviceInfoCollector* device_info_collector)
{
        int finish_value = 0;
        char finish_sem_file_name[MAX_FILE_NAME_LEN];
        sem_t* sem;
        get_finish_sem_file_name(finish_sem_file_name,
                device_info_collector->id);
        sem = sem_open(finish_sem_file_name, O_CREAT, S_IRWXU, finish_value);
        if(sem == SEM_FAILED)sys_panic("sem_open");
        sem_unlink(finish_sem_file_name);
        device_info_collector->finish_sem = sem;
        return 0;
}

static int _device_info_collector_init(
        DeviceInfoCollector* device_info_collector, int device)
{
        device_info_collector->id = device;
        return 0;
}

int device_info_collector_init(DeviceInfoCollector* device_info_collector,
                               int device)
{
        _device_info_collector_init(device_info_collector, device);
        device_info_collector_init_request_sem(device_info_collector);
        device_info_collector_init_finish_sem(device_info_collector);
        device_info_collector_init_daemon(device_info_collector);
        return 0;
}

int device_info_collector_collect(DeviceInfoCollector* device_info_collector)
{
        device_info_collector_collect_async(device_info_collector);
        device_info_collector_wait(device_info_collector);
        return 0;
}
        
int device_info_collector_collect_async(
        DeviceInfoCollector* device_info_collector)
{
        int err;
        err = sem_post(device_info_collector->request_sem);
        return err;
}

int device_info_collector_wait(DeviceInfoCollector* device_info_collector)
{
        int err;
        err = sem_wait(device_info_collector->finish_sem);
        return err;
}

int device_info_collector_destroy(DeviceInfoCollector* device_info_collector)
{
        cuCtxDetach(device_info_collector->cu_ctx);
        sem_close(device_info_collector->request_sem);
        sem_close(device_info_collector->finish_sem);
        return 0;
}
