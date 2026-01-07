CC=clang
ASM=nasm
CFLAGS=-O3 -Wall -Wextra
ASMFLAGS=-f elf64

all: wave_sort_asm

wave_sort.o: wave_sort.asm
	$(ASM) $(ASMFLAGS) wave_sort.asm -o wave_sort.o

wavesort_asm.o: wavesort_asm.c
	$(CC) $(CFLAGS) -c wavesort_asm.c -o wavesort_asm.o

wave_sort_asm: wavesort_asm.o wave_sort.o
	$(CC) $(CFLAGS) wavesort_asm.o wave_sort.o -o wave_sort_asm

clean:
	rm -f *.o wave_sort_asm
