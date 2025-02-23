# Build the image
build:
	docker build -t smart-attendance-system .

# Test the webcam
test:
	docker run --rm -it \
	--device=/dev/video0:/dev/video0 \
	smart-attendance-system \
	python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Webcam test:', cap.isOpened())"

# Run the container
run:
	docker run -p 5000:5000 \
    --network=host \
	--device=/dev/video0:/dev/video0 \
	-v $(PWD)/Attendance:/app/Attendance \
	-v $(PWD)/static/faces:/app/static/faces \
	-e DISPLAY=$(DISPLAY) \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	smart-attendance-system

push:
	docker tag smart-attendance-system:latest asifmahmoud414/smart-attendance-system:v1.0.1
	docker push asifmahmoud414/smart-attendance-system:v1.0.1
