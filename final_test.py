from ultralytics import YOLO
import cv2
import torch

def main():
    # Load the trained model
    model = YOLO("runs/detect/train2/weights/best.pt")

    # Ensure it uses CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"🖥️ Running on: {device}")

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        return

    print("🎥 Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference
        results = model.predict(source=frame, imgsz=640, conf=0.5, verbose=False)

        # Draw boxes and labels
        annotated_frame = results[0].plot()

        # Display result
        cv2.imshow("YOLOv8 Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
