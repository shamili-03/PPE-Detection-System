# PPE-Detection-System
**Project Title: PPE Detection and Alert System**

This project is a safety monitoring application designed to detect whether individuals in a video are wearing the necessary Personal Protective Equipment (PPE). The system uses advanced object detection techniques to identify PPE items such as hard hats, masks, and safety vests. It also detects non-compliance, such as the absence of these protective items. Upon detecting a violation, the system captures an image of the unsafe individual and sends an email alert with the image attached. Additionally, the system generates a pie chart to visually represent the total number of people detected, including those who are safe and unsafe.
This application is ideal for construction sites, manufacturing units, and other industrial environments where PPE compliance is critical to ensure workplace safety.

**Tools and Technologies Used**

YOLOv8:
Used for real-time object detection of PPE items.
Provides accurate detection of safety compliance and violations.

OpenCV:
Used for video processing and frame analysis.
Facilitates drawing bounding boxes and labels around detected objects.

Flask:
Acts as the backend framework for integrating the application with web interfaces.
Enables easy deployment and interaction through a web-based GUI.

Matplotlib:
Generates pie charts to visualize the compliance statistics (safe vs. unsafe persons).

Email Module (smtplib):
Sends email alerts when violations are detected.
Ensures timely communication of safety breaches to supervisors or managers.

Python:
The core programming language used for scripting the application.
Powers the integration of all tools and frameworks.

YOLO Weights (PPE Detection Model):
Pre-trained model fine-tuned for detecting PPE items like hard hats, masks, and safety vests.

Video Input (OpenCV Capture):
Processes pre-recorded or live video streams for analysis.
