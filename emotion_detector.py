import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define emotion the eotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

prev_frame = None

def detect_emotion(frame):
    global prev_frame
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        face_roi = gray[y:y+h, x:x+w]
        
        # Analyze facial features
        avg_brightness = np.mean(face_roi)
        contrast = np.std(face_roi)
        
        left_half = face_roi[:, :w//2]
        right_half = face_roi[:, w//2:]
        symmetry = np.abs(np.mean(left_half) - np.mean(right_half))
        movement = 0
        if prev_frame is not None:
            prev_roi = prev_frame[y:y+h, x:x+w]
            if prev_roi.shape == face_roi.shape:
                frame_diff = cv2.absdiff(face_roi, prev_roi)
                movement = np.mean(frame_diff)
        
        print(f"Brightness: {avg_brightness:.1f}, Contrast: {contrast:.1f}, "
              f"Symmetry: {symmetry:.2f}, Movement: {movement:.2f}")
        
        # Determine emotion based on thresholds
        if avg_brightness > 110 and contrast > 30 and contrast < 40:
            emotion = "Happy"
        elif movement > 8:
            emotion = "Surprise"
        elif contrast > 38:
            emotion = "Surprise"
        elif symmetry > 13: 
            emotion = "Angry"
        elif avg_brightness < 100:
            emotion = "Sad"
        else:
            emotion = "Neutral"
        
        confidence = 75
        
        cv2.putText(frame, f"{emotion} ({int(confidence)}%)", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    prev_frame = gray.copy()
    return frame

def main():
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect emotions
        result_frame = detect_emotion(frame)
        
        # Display the frame
        cv2.imshow('Emotion Detection', result_frame)
        
        # Exits the software
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
