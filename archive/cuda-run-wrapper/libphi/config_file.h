#ifndef _CONFIG_FILE_H_
#define _CONFIG_FILE_H_
/**
 * @file   config_file.h
 * @author xi huafeng <huafengxi@gmail.com>
 * @date   Mon Jul 20 10:04:27 2009
 * 
 * @brief  define class ConfigFile which can parse 'key=val' type confi file.
 *
 * @ingroup libphi
 * 
 */

#define CONFIG_FILE_MAX_ITEMS 32
#define CONFIG_FILE_MAX_NAME_LEN 128
#define CONFIG_FILE_MAX_LEN 1024

struct ConfigFile
{
        char file[CONFIG_FILE_MAX_NAME_LEN];
        char file_buf[CONFIG_FILE_MAX_LEN];
        char dict[CONFIG_FILE_MAX_ITEMS][2][CONFIG_FILE_MAX_NAME_LEN];
        int n_item;
};
typedef struct ConfigFile ConfigFile;

int config_file_init(ConfigFile* config_file, const char* file);
int config_file_read(ConfigFile* config_file);
int config_file_write(ConfigFile* config_file);
char* config_file_get(ConfigFile* config_file, const char* key);
int config_file_put(ConfigFile* config_file, const char* key, const char* val);
void config_file_print(ConfigFile* config_file);

#endif /* _CONFIG_FILE_H_ */
