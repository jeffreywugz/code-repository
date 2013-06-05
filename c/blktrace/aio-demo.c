#include<stdio.h>
#include<fcntl.h>
#include<string.h>
#include<stdlib.h>
#include<libaio.h>
#include<errno.h>
#include<unistd.h>
#include <linux/aio_abi.h>

int main(int argc, char** argv)
{
  int err = 0;
  int output_fd = 0;
  const char *content="hello world!";
  const char *outputfile="hello.txt";
  io_context_t ctx;
  struct iocb io,*p=&io;
  struct io_event e;
  struct timespec timeout;
  memset(&ctx,0,sizeof(ctx));
  if(0 != (err = io_setup(10, &ctx)))
  {
    error("io_setup error %s", strerror(errno));
  }
  else if((fd = open(file_name, O_CREAT|O_WRONLY, 0644)) < 0)
  {
    perror("open error");
    io_destroy(ctx);
    return -1;
  }
  io_prep_pwrite(&io, fd, content, strlen(content), 0);
  io.data=content;
  if(io_submit(ctx, 1, &p) != 1)
  {
    io_destroy(ctx);
    error("io_submit error\n");
    return -1;
  }
  while(1)
  {
    timeout.tv_sec=0;
    timeout.tv_nsec=500000000;//0.5s
    if(io_getevents(ctx, 0, 1, &e, &timeout) == 1)
    {
      close(output_fd);
      break;
    }
    printf("haven't done\n");
    sleep(1);
  }
  io_destroy(ctx);
  return 0;
}
