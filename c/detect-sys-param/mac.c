#include <sys/ioctl.h>
#include <net/if.h>
#include <unistd.h>
#include <netinet/in.h>
#include <string.h>
#include <stdio.h>

int64_t get_mac_addr(int sock, const char* name)
{
  int err = 0;
  struct ifreq ifr;
  int64_t mac = 0;
  if (NULL == name)
  {}
  else
  {
    strncpy(ifr.ifr_name, name, sizeof(ifr.ifr_name));
    ifr.ifr_name[sizeof(ifr.ifr_name)-1] = 0;
    if (0 != (err = ioctl(sock, SIOCGIFFLAGS, &ifr)))
    {
      perror("ioctl(GET IFFLAGS):");
    }
    else if (0 != (err = ioctl(sock, SIOCGIFHWADDR, &ifr)))
    {
      perror("ioctl(GET HWADDR):");
    }
    else
    {
      memcpy(&mac, ifr.ifr_hwaddr.sa_data, 6);
    }
  }
  return mac;
}

const char* get_default_ifname(int sock, char* buf, int64_t len)
{
  int err = 0;
  const char* ifname = NULL;
  struct ifconf ifc;
  ifc.ifc_len = len;
  ifc.ifc_buf = buf;

  if (0 != (err = ioctl(sock, SIOCGIFCONF, &ifc)))
  {
    perror("ioctl(get ifconf):");
  }
  else
  {
    struct ifreq ifr;
    struct ifreq* it = ifc.ifc_req;
    const struct ifreq* const end = it + (ifc.ifc_len / sizeof(*it));
    for (; 0 == err && it != end; ++it)
    {
      strcpy(ifr.ifr_name, it->ifr_name);
      if (0 != (err = ioctl(sock, SIOCGIFFLAGS, &ifr)))
      {
        perror("ioctl(get ifflags):");
      }
      else if (ifr.ifr_flags & IFF_LOOPBACK)
      {
        continue;
      }
      else
      {
        ifname = it->ifr_name;
        printf("ifname=%s\n", ifname);
      }
    }
  }
  return ifname;
}

int64_t get_default_mac()
{
  int64_t mac = 0;
  int sock = 0;
  char buf[1024];
  const char* ifname = NULL;
  if (0 >= (sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP)))
  {
    perror("socket():");
  }
  else if (NULL == (ifname = get_default_ifname(sock, buf, sizeof(buf))))
  {
  }
  else if (mac = get_mac_addr(sock, ifname))
  {
  }
  if (sock >= 0)
  {
    close(sock);
  }
  return mac;
}

int main()
{
  printf("get_default_mac()=>%lx", get_default_mac());
  return 0;
}
