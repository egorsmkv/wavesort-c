CC=clang
ASM=nasm
CFLAGS=-O3 -Wall -Wextra
ASMFLAGS=-f elf64

all: wavesort_asm

wavesort.o: wavesort.asm
	$(ASM) $(ASMFLAGS) wavesort.asm -o wavesort.o

wavesort_asm.o: wavesort_asm.c
	$(CC) $(CFLAGS) -c wavesort_asm.c -o wavesort_asm.o

wavesort_asm: wavesort_asm.o wave_sort.o
	$(CC) $(CFLAGS) wavesort_asm.o wavesort.o -o wavesort_asm

clean:
	rm -f *.o wavesort_asm
