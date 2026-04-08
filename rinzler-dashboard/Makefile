build:
	go build -o rinzler-dashboard .

install: build
	cp rinzler-dashboard $(GOBIN)/rinzler-dashboard

clean:
	rm -f rinzler-dashboard

.PHONY: build install clean
