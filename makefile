flags=-lcublas
filename=testCuda

all:
	nvcc ${filename}.cu ${flags} -o ${filename}
