CC = mpicc

me471 = /home/brenton/BSU/ME471

SRC = ${me471}/ME471-571/src
INCLUDE_DIR = -I${me471}/ME471-571/src -I/usr/include -I${me471}/Homework_3

objs = ${me471}/Homework_3/demo_util.c

.PHONY: hmwk3_1c
 
hmwk3_1c : hmwk3_1c.c ${objs}
	${CC} -o hmwk3_1c  ${INCLUDE_DIR} hmwk3_1c.c ${objs} -lm

hmwk3_2c : hmwk3_2c.c ${objs}
	${CC} -o hmwk3_2c  ${INCLUDE_DIR} hmwk3_2c.c ${objs} -lm

hmwk3_3c : hmwk3_3c.c ${objs}
	${CC} -o hmwk3_3c  ${INCLUDE_DIR} hmwk3_3c.c ${objs} -lm

hmwk3_1n : hmwk3_1n.c ${objs}
	${CC} -o hmwk3_1n  ${INCLUDE_DIR} hmwk3_1n.c ${objs} -lm

hmwk3_2n : hmwk3_2n.c ${objs}
	${CC} -o hmwk3_2n  ${INCLUDE_DIR} hmwk3_2n.c ${objs} -lm
