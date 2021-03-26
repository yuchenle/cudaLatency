CC = nvcc
all: sender pluser

sender: sender_driver.cu Sender.cu
	$(CC) $^ -o $@

pluser: pluser_driver.cu Pluser.cu
	$(CC) $^ -o $@

clean:
	rm sender pluser
