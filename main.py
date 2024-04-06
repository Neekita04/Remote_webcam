import cv2

def main():
    # URL to the phone camera feed (replace with your phone's IP address)
    url = "http://192.168.171.56:8080/video"

    # Initialize the video stream
    camera = cv2.VideoCapture(url)

    # Set the window size
    window_width = 640
    window_height = 480
    cv2.namedWindow('Phone Camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Phone Camera', window_width, window_height)

    image_count = 1

    while True:
        # Read the next frame from the camera
        ret, frame = camera.read()

        # Display the frame
        cv2.imshow('Phone Camera', frame)

        # Press 's' to save the frame as an image
        if cv2.waitKey(1) & 0xFF == ord('s'):
            file_name = 'captured_image_{}.jpg'.format(image_count)
            cv2.imwrite(file_name, frame)
            print("Image {} captured and saved as {}".format(image_count, file_name))
            image_count += 1

        # Press 'q' to quit
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close OpenCV windows
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

'''
import cv2

def main():
    # URL to the phone camera feed (replace with your phone's IP address)
    url = "http://192.168.171.56:8080/video"

    # Initialize the video stream       
    camera = cv2.VideoCapture(url)

    # Set the window size
    window_width = 640
    window_height = 480
    cv2.namedWindow('Phone Camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Phone Camera', window_width, window_height)

    image_count = 1

    # Load the pre-trained face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        # Read the next frame from the camera
        ret, frame = camera.read()

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Phone Camera', frame)

        # Press 's' to save the frame as an image
        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            file_name = 'captured_image_{}.jpg'.format(image_count)
            cv2.imwrite(file_name, frame)
            print("Image {} captured and saved as {}".format(image_count, file_name))
            image_count += 1

        # Press 'q' to quit
        elif key & 0xFF == ord('q'):
            break
    
    # Release the camera and close OpenCV windows
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
'''