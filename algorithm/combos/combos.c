#include <stdbool.h>
#include <stdio.h>
#include <string.h>

struct Arranger {
        int num_result;
        int cur_idx_arrange;
        int arrangement[9];
        int numbers[10];
};

void arranger_init(struct Arranger* arranger, int* numbers)
{
        int i;
        arranger->num_result = 0;
        arranger->cur_idx_arrange = 0;
        memset(arranger->numbers, 0, sizeof(arranger->numbers));
        for(i = 0; i < 9; i++)
                arranger->numbers[numbers[i]]++;
}

bool arranger_check_result(struct Arranger* arranger)
{
        int* result = arranger->arrangement;
        int row1 = result[0] + result[1] + result[2];
        int row2 = result[3] + result[4] + result[5];
        int row3 = result[6] + result[7] + result[8];
        int col1 = result[0] + result[3] + result[6];
        int col2 = result[1] + result[4] + result[7];
        int col3 = result[2] + result[5] + result[8];
        return row1 == row2 && row1 == row3 && col1 == col2 && col1 == col3;
}

void arranger_arrange(struct Arranger* arranger)
{
        int i;
        if(arranger->cur_idx_arrange == 9){
                if(arranger_check_result(arranger))
                        arranger->num_result++;
                return;
        }
        for(i = 0; i < 10; i++){
                if(arranger->numbers[i] == 0)
                        continue;
                arranger->numbers[i]--;
                arranger->arrangement[arranger->cur_idx_arrange] = i;
                arranger->cur_idx_arrange++;
                arranger_arrange(arranger);
                arranger->numbers[i]++;
                arranger->cur_idx_arrange--;
        }
}

int arrange_get_num_result(struct Arranger* arranger)
{
        return arranger->num_result;
}

#define arrary_len(x) (sizeof(x)/sizeof(x[0]))
int main()
{
        int i;
        struct Arranger arranger;
        int numbers[][9] = {
                {1,2,3,3,2,1,2,2,2},
                {4,4,4,4,4,4,4,4,4},
                {1,5,1,2,5,6,2,3,2},
                {1,2,6,6,6,4,2,6,4},
        };

        for(i = 0; i < arrary_len(numbers); i++){
                arranger_init(&arranger, numbers[i]);
                arranger_arrange(&arranger);
                printf("%d\n", arrange_get_num_result(&arranger));
        }
	return 0;
}
