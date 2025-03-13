import cv2
import os

# Define folder mapping for screw types
screw_folders = {
    "1": "Screwtype1",
    "2": "Screwtype2",
    "3": "Screwtype3",
    "4": "Screwtype4"
}

# Check if folders exist, if not, create them
for folder in screw_folders.values():
    os.makedirs(folder, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)  # Change index if needed

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Press keys [1-4] to save images in respective screw folders.")
print("Press 'q' to exit.")

# Count existing images for proper numbering
image_counts = {key: len(os.listdir(folder)) for key, folder in screw_folders.items()}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # Display screw counts on the frame
    y_offset = 30
    for key, folder in screw_folders.items():
        cv2.putText(frame, f"{folder}: {image_counts[key]}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30

    cv2.imshow("Screw Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break  # Quit the script

    elif chr(key) in screw_folders:
        folder = screw_folders[chr(key)]
        image_path = os.path.join(folder, f"{folder}_{image_counts[chr(key)] + 1}.jpg")
        cv2.imwrite(image_path, frame)
        image_counts[chr(key)] += 1
        print(f"Saved: {image_path}")

cap.release()
cv2.destroyAllWindows()
