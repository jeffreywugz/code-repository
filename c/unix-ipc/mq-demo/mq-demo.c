#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <mqueue.h>


#define MAX_MSG_LEN 256

void panic(const char* msg)
{
        perror(msg);
        exit(-1);
}

int main()
{
        pid_t pid;
        mqd_t mqd;
        const char* mq_name = "/file.mq";
        char msg_send_buf[] = "hello...";
        char msg_recv_buf[MAX_MSG_LEN];
        int msg_len = sizeof(msg_send_buf);
        int msg_prio = 0;
        struct mq_attr mq_attr;


        mqd = mq_open(mq_name, O_RDWR | O_CREAT, S_IRWXU, NULL);
        if(mqd < 0)panic("mq_open");
        unlink(mq_name);
        
        pid = fork();
        if(pid < 0)panic("fork");
        if(pid >0){
                if(mq_send(mqd, msg_send_buf, msg_len, msg_prio)<0)
                        panic("mq_send");
                
        } else {
                if(mq_getattr(mqd, &mq_attr)<0)panic("mq_getattr");
                if(mq_receive(mqd, msg_recv_buf, mq_attr.mq_msgsize, NULL)<0)
                        panic("mq_recv");
                printf("%s\n", msg_recv_buf);
        }
        
        mq_close(mqd);
	return 0;
}
