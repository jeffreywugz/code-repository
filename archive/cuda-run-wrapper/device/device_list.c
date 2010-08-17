#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <unistd.h>
#include "device_list.h"
#include "device_evaluator.h"
#include "util.h"

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err){
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(-1);
    }                         
}

int device_list_init(DeviceList* device_list)
{
        int i;
        cudaGetDeviceCount(&device_list->n_device);
        for(i=0; i<device_list->n_device; i++){
                device_init(&device_list->devices[i], i);
#ifdef DAEMON
                device_info_collector_init(
                        &device_list->device_info_collector[i], i);
#endif
        }
        return 0;
}

void device_list_print(DeviceList* device_list)
{
        int i;
        printf("device list information:\n");
        for(i=0; i<device_list->n_device; i++)
                device_print(&device_list->devices[i]);
}

#ifdef DAEMON
int device_list_get_info(DeviceList* device_list, DeviceInfo* device_info)
{
        int i;
        
        for(i=0; i<device_list->n_device; i++){
                device_info_collector_collect_async(
                        &device_list->device_info_collector[i]);
        }
        
        for(i=0; i<device_list->n_device; i++){
                device_info_collector_wait(
                        &device_list->device_info_collector[i]);
        }
        
        for(i=0; i<device_list->n_device; i++){
                memcpy(&device_info[i],
                       &device_list->device_info_collector[i].device_info,
                       sizeof(DeviceInfo));
        }
        
        return 0;
}
#endif

void device_list_set_info(DeviceList* device_list, DeviceInfo* device_info)
{
        int i;
        for(i=0; i<device_list->n_device; i++){
                device_list->devices[i].info = device_info[i];
        }
}

static void device_list_sort(DeviceList* device_list)
{
        int i, j;
        int n_device = device_list->n_device;
        Device* devices = device_list->devices;
        for(i = n_device-1; i > 0; i--)
                for(j = 0; j < i; j++){
                        if(devices[j].score < devices[j+1].score)
                                swap(devices[j], devices[j+1]);
                }
}

static void device_list_evaluate(DeviceList* device_list)
{
        DeviceEvaluator device_evaluator;
        device_evaluator_init(&device_evaluator, device_list->n_device,
                              device_list->devices);
        device_evaluator_eval(&device_evaluator);
}

static void _device_list_set_best(DeviceList* device_list)
{
        int i;
        int device_id_list[MAX_DEVICE_COUNT];
        for(i = 0; i < device_list->n_device; i++)
                device_id_list[i] = device_list->devices[i].id;
        cudaSetValidDevices(device_id_list, device_list->n_device);
}

void device_list_set_best(DeviceList* device_list)
{
        device_list_evaluate(device_list);
        device_list_sort(device_list);
        _dbg(device_list_print(device_list));
        _device_list_set_best(device_list);
}

int device_list_destroy(DeviceList* device_list)
{
        return 0;
}
