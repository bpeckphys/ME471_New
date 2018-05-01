CC = mpicc

me471 = /home/brenton/BSU/ME471

SRC = ${me471}/ME471-571/src
INCLUDE_DIR = -I${me471}/ME471-571/src -I/usr/include -I${me471}/Final_Project

objs = ${me471}/Final_Project/demo_util.c

.PHONY: Final_Project
 
Final_Project : Final_Project.c ${objs}
	${CC} -o Final_Project  ${INCLUDE_DIR} Final_Project.c ${objs} -lm
	
Final_Project2 : Final_Project2.c ${objs}
	${CC} -o Final_Project2  ${INCLUDE_DIR} Final_Project2.c ${objs} -lm
	
Final_Project3 : Final_Project3.c ${objs}
	${CC} -o Final_Project3  ${INCLUDE_DIR} Final_Project3.c ${objs} -lm
