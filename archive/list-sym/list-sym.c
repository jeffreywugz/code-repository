#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <libelf.h>

static void panic(const char* msg)
{
        fprintf(stderr, msg);
        exit(-1);
}

#define MAX_N_SYM 128
struct ElfSym
{
        const char* name;
        void* addr;
};
typedef struct ElfSym ElfSym;

struct ElfFile
{
        const char* file_name;
        int fd;
        Elf* elf;
        int n_func;
        int n_obj;
        ElfSym func[MAX_N_SYM];
        ElfSym obj[MAX_N_SYM];
};
typedef struct ElfFile ElfFile;

static int elf_file_init_elf(ElfFile* elf_file, const char* file_name)
{
        elf_file->file_name = file_name;
        
        elf_file->fd = open(elf_file->file_name, O_RDONLY);
        if(elf_file->fd < 0)panic("can not open file!\n");
        
        if(elf_version(EV_CURRENT) != EV_CURRENT)
                panic("version not compatible!\n");
        elf_file->elf = elf_begin(elf_file->fd, ELF_C_READ, NULL);
        if(elf_file->elf == NULL)panic(elf_errmsg(elf_errno()));
        return 0;
}

static int elf_file_fill_dynsymtab(ElfFile* elf_file,
                                   Elf_Scn* scn, Elf32_Shdr* shdr)
{
        Elf_Data* data = elf_getdata(scn, NULL);
        Elf32_Sym* sym = data->d_buf;
        Elf32_Sym *lastsym = (Elf32_Sym*) ((char*) data->d_buf + data->d_size);
        char* name;
        int n_func = 0, n_obj = 0;
        int i;
        for(i = 0; sym < lastsym; sym++, i++){
                name = elf_strptr(elf_file->elf,
                                  shdr->sh_link , (size_t)sym->st_name);
                if(ELF32_ST_TYPE(sym->st_info) == STT_OBJECT){ 
                        elf_file->obj[n_obj].name = name;
                        elf_file->obj[n_obj].addr = (void*)sym->st_value;
                        n_obj++;
                }
                if(ELF32_ST_TYPE(sym->st_info) == STT_FUNC){ 
                        elf_file->func[n_func].name = name;
                        elf_file->func[n_func].addr = (void*)sym->st_value;
                        n_func++;
                }
        }
        elf_file->n_func = n_func;
        elf_file->n_obj = n_obj;
        return 0;
}

static int elf_file_init_dynsymtab(ElfFile* elf_file)
{
        Elf_Scn* scn = NULL;
        Elf32_Shdr* shdr;
        
        while ((scn= elf_nextscn(elf_file->elf, scn)) != NULL){
                shdr = elf32_getshdr(scn);
                if(shdr->sh_type == SHT_DYNSYM)
                        break;
        }
        if(scn == NULL)return -1;
        elf_file_fill_dynsymtab(elf_file, scn, shdr);
        return 0;
}

void elf_file_print(ElfFile* elf_file)
{
        int i;
        
        printf("num of dynamic functions = %d\n", elf_file->n_func);
        for(i = 0; i < elf_file->n_func; i++){
                printf("%s: %p\n",
                       elf_file->func[i].name, elf_file->func[i].addr);
        }
        
        printf("num of dynamic objects = %d\n", elf_file->n_obj);
        for(i = 0; i < elf_file->n_obj; i++){
                printf("%s: %p\n",
                       elf_file->obj[i].name, elf_file->obj[i].addr);
        }
}

int elf_file_init(ElfFile* elf_file, const char* file_name)
{
        elf_file_init_elf(elf_file, file_name);
        elf_file_init_dynsymtab(elf_file);
        return 0;
}

int elf_file_destroy(ElfFile* elf_file)
{
        elf_end(elf_file->elf);
        close(elf_file->fd);
	return 0;
}

int main(int argc, char* argv[])
{
        ElfFile elf_file;
        char exe_file[256];
        
        realpath(argv[0], exe_file);
        elf_file_init(&elf_file, exe_file);
        elf_file_print(&elf_file);
        elf_file_destroy(&elf_file);

	return 0;
}
