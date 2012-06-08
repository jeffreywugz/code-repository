for i in {1..16}; do
	./epoll.exe client 10.232.35.40 8008 1 100000;
done;
wait;
