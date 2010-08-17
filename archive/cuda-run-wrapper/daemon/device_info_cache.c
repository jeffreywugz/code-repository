#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/signal.h>
#include "util.h"
#include "device_info_cache.h"


static int device_info_cache_is_valid(DeviceInfoCache* device_info_cache)
{
        long long time_stamp;
        long long time_passed;
        time_stamp = get_time();
        time_passed = time_stamp - device_info_cache->time_stamp;
        return time_passed < device_info_cache->max_cache_time_interval
                && device_info_cache->cache_requests < device_info_cache->max_cache_requests;
}

static void device_info_cache_copy_out(DeviceInfoCache* device_info_cache,
                                            DeviceInfo* device_info)
{
        int len = device_info_cache->device_list.n_device * sizeof(DeviceInfo);
        pthread_mutex_lock(&device_info_cache->device_info_lock);
        device_info_cache->cache_requests++;
        memcpy(device_info, device_info_cache->device_info, len);
        pthread_mutex_unlock(&device_info_cache->device_info_lock);
}

static void device_info_cache_copy_in(DeviceInfoCache* device_info_cache,
                                      DeviceInfo* device_info)
{
        int len = device_info_cache->device_list.n_device * sizeof(DeviceInfo);
        pthread_mutex_lock(&device_info_cache->device_info_lock);
        device_info_cache->time_stamp = get_time();
        device_info_cache->cache_requests = 0;
        memcpy(device_info_cache->device_info, device_info, len);
        pthread_mutex_unlock(&device_info_cache->device_info_lock);
}

static void _device_info_cache_update(DeviceInfoCache* device_info_cache)
{
        DeviceInfo device_info[MAX_DEVICE_COUNT];
        debug("_device_info_cache_update\n");
        device_list_get_info(&device_info_cache->device_list, device_info);
        device_info_cache_copy_in(device_info_cache, device_info);
}

static void device_info_cache_update(DeviceInfoCache* device_info_cache)
{
        pthread_kill(device_info_cache->update_thread, SIGALRM);
}

static void device_info_cache_update_thread_handle(
        DeviceInfoCache* device_info_cache)
{
        while(1){
                pause();
                _device_info_cache_update(device_info_cache);
        }
}

static void sigalrm_handler(int signo)
{
}

static void device_info_cache_update_thread_init(
        DeviceInfoCache* device_info_cache)
{
        int err;
        struct sigaction actions;

        memset(&actions, 0, sizeof(actions));
        sigemptyset(&actions.sa_mask);
        actions.sa_flags = 0;
        actions.sa_handler = sigalrm_handler;

        err = sigaction(SIGALRM, &actions, NULL);
        if(err < 0)panic("sigaction");
        
        err = pthread_create(&device_info_cache->update_thread, NULL,
                             (void*)device_info_cache_update_thread_handle,
                             (void*)device_info_cache);
        if(err < 0)sys_panic("pthread_create");
}

int device_info_cache_init(DeviceInfoCache* device_info_cache,
        long long max_cache_time_interval, int max_cache_requests)
{
        device_list_init(&device_info_cache->device_list);
        pthread_mutex_init(&device_info_cache->device_info_lock, NULL);
        device_info_cache_update_thread_init(device_info_cache);
        device_info_cache->time_stamp = 0;
        device_info_cache->max_cache_time_interval = max_cache_time_interval;
        device_info_cache->max_cache_requests = max_cache_requests;
        /* invalid first */
        device_info_cache->cache_requests = max_cache_requests + 1; 
        return 0;
}

int device_info_cache_get(DeviceInfoCache* device_info_cache,
                          DeviceInfo* device_info)
{
        if(!device_info_cache_is_valid(device_info_cache)){
                device_info_cache_update(device_info_cache);
        }
        device_info_cache_copy_out(device_info_cache, device_info);
        return 0;
}

int device_info_cache_destroy(DeviceInfoCache* device_info_cache)
{
        pthread_mutex_destroy(&device_info_cache->device_info_lock);
        return 0;
}
