import cv2
import cvzone
import math
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from ultralytics import YOLO
import os

def ppe_detection(file):
    if file is None:
        cap = cv2.VideoCapture(0)  # For Webcam
        cap.set(3, 1280)
        cap.set(4, 720)
    else:
        cap = cv2.VideoCapture(file)  # For Video

    model = YOLO("best.pt")
    



    classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
                  'Safety Vest', 'machinery', 'vehicle']
    
    # Initialize a DataFrame to store the counts
    class_counts_df = pd.DataFrame(columns=['Frame', 'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 
                                            'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 
                                            'machinery', 'vehicle'])

    frame_count = 0
    max_frames = 5  # Limit to 100 frames

    while True:
        success, img = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count > max_frames:
            break  # Stop after 100 frames

        # Process the image using YOLO
        results = model(img, stream=True)
        
        # Initialize a dictionary to store counts for this frame
        frame_counts = {className: 0 for className in classNames}

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                currentClass = classNames[cls]
                
                if conf > 0.5:
                    # Update the count for this class
                    frame_counts[currentClass] += 1

                    # Drawing rectangles and adding text (optional)
                    if currentClass in ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']:
                        myColor = (0, 0, 255)  # red
                    elif currentClass in ['Hardhat', 'Safety Vest', 'Mask']:
                        myColor = (0, 255, 0)  # green
                    else:
                        myColor = (255, 0, 0)  # blue

                    cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                                       scale=1, thickness=1, colorB=myColor, colorT=(255, 255, 255),
                                       colorR=myColor, offset=5)
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

        # Display summary for this frame
        summary_str = ", ".join([f"{frame_counts[className]} {className}s" for className in classNames])
        print(summary_str)

        # Append the results to the DataFrame for this frame
        class_counts_df.loc[frame_count] = [frame_count] + list(frame_counts.values())
        
        # Show the image with the bounding boxes and labels
        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save the dataframe to a CSV file
    file_path = r"C:\Users\hp\Desktop\ppe_detection_summary.csv"  # Modify this path if needed
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure the directory exists
    class_counts_df.to_csv(file_path, index=False)
    print(f"Data saved to: {file_path}")
    
    # Now, let's create the pie chart based on the total count for each class.
    total_counts = class_counts_df.drop(columns='Frame').sum()

    # Generate pie chart for the class counts
    plt.figure(figsize=(8, 8))
    plt.pie(total_counts, labels=total_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    plt.title('Class Distribution for PPE Detection')

    # Save the pie chart as an image
    pie_chart_path = "ppe_detection_pie_chart.png"
    plt.savefig(pie_chart_path)
    plt.close()

    # Create a new Excel workbook to save the data and the pie chart
    wb = Workbook()
    ws = wb.active
    ws.title = "Detection Summary"
    
    # Write the DataFrame to the Excel sheet
    for r in dataframe_to_rows(class_counts_df, index=False, header=True):
        ws.append(r)

    # Insert the pie chart image into the Excel sheet
    img = Image(pie_chart_path)
    ws.add_image(img, 'L5')  # Place the image starting from cell L5

    # Save the Excel file
    excel_file_path = r"C:\Users\hp\Desktop\ppe_detection_with_pie_chart.xlsx"
    wb.save(excel_file_path)
    print(f"Excel file saved to: {excel_file_path}")
    
    # Clean up and delete the pie chart image after embedding it
    os.remove(pie_chart_path)

    cv2.destroyAllWindows()

def dataframe_to_rows(df, index=False, header=True):
    """Converts a pandas DataFrame to rows for inserting into openpyxl."""
    from openpyxl.utils.dataframe import dataframe_to_rows
    return dataframe_to_rows(df, index=index, header=header)

if __name__ == "__main__":
    # Example usage
    file = r"C:\Users\hp\Desktop\PPE_detection_YOLO\Videos\ppe-2.mp4"
    ppe_detection(file)