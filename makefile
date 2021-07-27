flags=-lcublas
filename=example1

all:
	nvcc ${filename}.cu ${flags} -o ${filename}
