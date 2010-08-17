#include "device_list.h"
#include "device_evaluator.h"

int device_evaluator_init(DeviceEvaluator* device_evaluator, int n_device,
                          Device* devices)
{
        device_evaluator->n_device = n_device;
        device_evaluator->devices = devices;
        return 0;
}

static void device_evaluator_eval_free_mem_only(
        DeviceEvaluator* device_evaluator)
{
        int i;
        Device* devices;
        devices = device_evaluator->devices;
        for(i=0; i < device_evaluator->n_device; i++){
                devices[i].score = devices[i].info.free_mem;
        }
}

struct DeviceScore
{
        double mem;
        double delay;
        double score;
};

static void device_score_normalize(int n_devices, Device* devices,
                                   struct DeviceScore* score)
{
        double mem_sum;
        double delay_sum;
        int i;

        for(mem_sum = 0, delay_sum =0, i=0; i < n_devices; i++){
                mem_sum += devices[i].info.free_mem + 1;
                delay_sum += devices[i].info.delay + 1;
        }

        for(i=0; i < n_devices; i++){
                score[i].mem = (devices[i].info.free_mem+1)/mem_sum;
                score[i].delay = (devices[i].info.delay+1)/delay_sum;
        }
}

static void compute_weighted_score(int n_device, struct DeviceScore* score)
{
        int i;
        const double mem_weight = 1.0;
        const double delay_weight = -0.2;
        for(i=0; i < n_device; i++){
                score[i].score = score[i].mem * mem_weight
                        + score[i].delay * delay_weight;
        }
}

static void device_score_copy_out(int n_device, struct DeviceScore* score,
                                  Device* devices)
{
        int i;
        for(i=0; i < n_device; i++){
                devices[i].score = score[i].score;
        }
}

static void device_evaluator_eval_default(
        DeviceEvaluator* device_evaluator)
{
        int n_device;
        Device* devices;
        struct DeviceScore score[MAX_DEVICE_COUNT];
        
        n_device = device_evaluator->n_device;
        devices = device_evaluator->devices;
        
        device_score_normalize(n_device, devices, score);
        compute_weighted_score(n_device, score);
        device_score_copy_out(n_device, score, devices);
}

void device_evaluator_eval(DeviceEvaluator* device_evaluator)
{
        device_evaluator_eval_default(device_evaluator);
}
