#include <stdbool.h>
#include <stdio.h>

// test to see if bss section is properly expanded 

static int mydata[1000000];

bool checkdata()
{
  printf("check data!\n");
	return ( mydata[500000] == 0 );
}
