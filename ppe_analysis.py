from ultralytics import YOLO
import cv2
import cvzone
import math
import smtplib
from email.message import EmailMessage
import time

# Set up email parameters
EMAIL_ADDRESS = "palapati.shamili33@gmail.com"
EMAIL_PASSWORD = "lktv tine pjdd sprp"
TO_EMAIL = "palapati.shamili03@gmail.com"

def send_email(image_path):
    msg = EmailMessage()
    msg['Subject'] = "PPE Violation Detected"
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = TO_EMAIL
    msg.set_content("A PPE violation was detected. See the attached image.")

    with open(image_path, 'rb') as f:
        img_data = f.read()
        msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename="ppe_violation.jpg")

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)

def ppe_detection(file):
    if file is None:
        cap = cv2.VideoCapture(0)  # Webcam
        cap.set(3, 1280)
        cap.set(4, 720)
    else:
        cap = cv2.VideoCapture(file)  # Video file

    model = YOLO("best.pt")
    classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
                  'Safety Vest', 'machinery', 'vehicle']
    myColor = (0, 0, 255)
    violation_count = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if conf > 0.5:
                    if currentClass in ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']:
                        myColor = (0, 0, 255)  # Red for violation
                        violation_count += 1
                        
                        # Capture and save image of violation
                        timestamp = int(time.time())
                        image_path = f"ppe_violation_{timestamp}.jpg"
                        cv2.imwrite(image_path, img)

                        # Send email with violation image
                        send_email(image_path)

                    elif currentClass in ['Hardhat', 'Safety Vest', 'Mask']:
                        myColor = (0, 255, 0)  # Green for safe

                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), 
                                       scale=1, thickness=1, colorB=myColor, colorT=(255, 255, 255),
                                       colorR=myColor, offset=5)
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    file = r"C:\Users\hp\Desktop\PPE_detection_YOLO\Videos\ppe-2.mp4"
    ppe_detection(file)
