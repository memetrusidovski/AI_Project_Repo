#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 16

void initialize(void);
void shuffle(void);
void show(void);
void loop(void);

int a[N];
int z; /* index of the piece 0 (= empty) */

int main()
{
    initialize();
    shuffle();
    loop();
}

void initialize(void)
{
    int i;

    for (i = 0; i < N; i++)
    {
        a[i] = i;
    }
}

void shuffle(void)
{ /* shuffle the pieces */
    int i;

    srand((unsigned)time(NULL));

    for (i = N; i > 0; i--)
    {
        int tmp;
        int r;

        r = rand() % i;
        tmp = a[i - 1];
        a[i - 1] = a[r];
        a[r] = tmp;
    }

    for (i = 0; i < N; i++)
    {
        if (a[i] == 0)
        {
            z = i;
        }
    }
}

void show(void)
{ /* print the array */
    int i;

    for (i = 0; i < N; i++)
    {
        if (a[i] == 0)
        {
            printf("   "); /* 0 means the empty square */
        }
        else
        {
            printf(" %2d", a[i]);
        }
    }
    putchar('\n');
}

void loop(void)
{
    int i;

    for (;;)
    {
        show();
        switch (z)
        {
        case 0:
        case 2:
        case 4:
        case 8:
        case 10:
        case 12:
            if (a[z + 1] != 1 && a[z + 2] != 1 && a[z + 3] != 1 && (a[z + 1] > a[z + 3] || a[z + 2] > a[z + 3]))
            {
                a[z] = a[z + 3], a[z + 3] = 0;
                z = z + 3;
            }
            else
            {
                a[z] = a[z + 1], a[z + 1] = 0;
                z = z + 1;
            }
            break;
        case 15:
            a[15] = a[0], a[0] = 0;
            z = 0;
            break;
        default:
            a[z] = a[z + 1], a[z + 1] = 0;
            z++;
        }

        for (i = 4; i < N && a[i] == i; i++)
        {
            ;
        }
        if (i == N && a[0] == 0 && a[1] == 1)
        {
            show();
            break;
        }
    }
}