import cv2
import os

# Define folder mapping for screw types
# Each key corresponds to a keyboard key, and the value is the folder name where images will be saved
base_path = "dataset_path"
screw_folders = {
    "1": os.path.join(base_path, "Screwtype1"),
    "2": os.path.join(base_path, "Screwtype2"),
    "3": os.path.join(base_path, "Screwtype3"),
    "4": os.path.join(base_path, "Screwtype4"),
    "5": os.path.join(base_path, "Noscrew1")
}

# Check if folders exist, if not, create them
for folder in screw_folders.values():
    os.makedirs(folder, exist_ok=True)  # Create folder if it doesn't already exist

# Open webcam
cap = cv2.VideoCapture(0)  # Open the default webcam (index 0). Change index if using a different camera.

# Check if the webcam was successfully opened
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()  # Exit the script if the webcam cannot be accessed

# Instructions for the user
print("Press keys [1-5] to save images in respective screw folders.")
print("Press 'q' to exit.")

# Count existing images in each folder for proper numbering
image_counts = {key: len(os.listdir(folder)) for key, folder in screw_folders.items()}

# Main loop for capturing and saving images
while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # Display screw counts on the frame
    y_offset = 30  # Vertical position for the first text line
    for key, folder in screw_folders.items():
        folder_name = os.path.basename(folder)  # Get the folder name (e.g., "Screwtype1")
        cv2.putText(frame, f"{folder_name}: {image_counts[key]}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Green text
        y_offset += 30  # Move to the next line for the next folder

    # Show the video feed with the overlay
    cv2.imshow("Screw Data Collection", frame)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF  # Get the ASCII value of the pressed key

    if key == ord('q'):  # If 'q' is pressed, exit the loop
        break

    elif chr(key) in screw_folders:  # Check if the pressed key corresponds to a folder
        folder = screw_folders[chr(key)]  # Get the folder path for the pressed key
        folder_name = os.path.basename(folder)  # Get the folder name (e.g., "Noscrew1")
        # Generate a unique file name for the new image
        image_path = os.path.join(folder, f"{folder_name}_{image_counts[chr(key)] + 1}.jpg")
        cv2.imwrite(image_path, frame)  # Save the current frame as an image
        image_counts[chr(key)] += 1  # Increment the image count for the corresponding folder
        print(f"Saved: {image_path}")  # Notify the user that the image was saved

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()