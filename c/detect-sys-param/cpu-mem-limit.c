#include "unistd.h"

int main()
{
  printf("cpu_total=%ld, cpu_online=%ld, page_size=%ld, phy_page=%ld, availiable_phy_page=%ld",
         sysconf(_SC_NPROCESSORS_CONF), sysconf(_SC_NPROCESSORS_ONLN),
         sysconf(_SC_PAGE_SIZE),
         sysconf(_SC_PHYS_PAGES),
         sysconf(_SC_AVPHYS_PAGES));
}
