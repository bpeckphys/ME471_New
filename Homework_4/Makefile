CC = mpicc

me471 = /home/brenton/BSU/ME471

INCLUDE_DIR = -I${me471}/ME471-571/src -I/usr/include -I${me471}/Homework_4

objs = ${me471}/ME471-571/src/demo_util.c

.PHONY: hmwk4_1
 
hmwk4_1c : hmwk4_1.c ${objs}
	${CC} -o hmwk4_1  ${INCLUDE_DIR} hmwk4_1.c ${objs} -lm

