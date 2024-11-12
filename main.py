# Import necessary libraries
from ultralytics import YOLO
import cv2
import numpy as np
import datetime


# Function to generate a unique video name based on the current date and time
def generate_video_name(output_video_path):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    extension = output_video_path.split('.')[-1]
    new_video_name = f"video_{timestamp}.{extension}"
    return new_video_name

# Angle Calculator
def calculate_angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    angle_rad = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    angle_deg = np.degrees(angle_rad)
    if not np.isnan(angle_deg):
        return int(angle_deg)
    else:
        return None

def kpxy(image, point):
    x, y = point[0], point[1]
    h, w = image.shape[:2]
    px, py = int(x * w), int(y * h)
    return [px, py]

def visual_keypoints(image, keypoints):
    for keypoint in keypoints:
        for i, (x, y) in enumerate(keypoint):
            h, w = image.shape[:2]
            px, py = int(x * w), int(y * h)
            cv2.circle(image, (px, py), radius=3, color=(0, 0, 255), thickness=-1)
            cv2.putText(image, str(i + 1), (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    return image

def add_text_top_left(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 0, 255), thickness=2):
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return image

def check_posture(image, keypoint, box, postures_to_check=None):
    if postures_to_check is None:
        print("Add body posture to check")

    def evaluate_posture_condition(image, p1, p2, p3, angle_name):
        angle = calculate_angle(p1, p2, p3)
        response_text = f"{angle}" if angle else "N/A"
        color = (0, 255, 0) if angle in range(90, 121) else (0, 0, 255)
        image = add_text_top_left(image, text=response_text, position=p2, color=color, font_scale=0.7)
        return image
    if len(keypoint) > 16:
        if "back" in postures_to_check:
            image = evaluate_posture_condition(image, kpxy(image, keypoint[6]), kpxy(image, keypoint[12]), kpxy(image, keypoint[14]), "BA")
        if "shoulder" in postures_to_check:
            image = evaluate_posture_condition(image, kpxy(image, keypoint[6]), kpxy(image, keypoint[8]), kpxy(image, keypoint[10]), "AA")
        if "leg" in postures_to_check:
            image = evaluate_posture_condition(image, kpxy(image, keypoint[12]), kpxy(image, keypoint[14]), kpxy(image, keypoint[16]), "leg")
    else:
        print("Insufficient keypoints detected to check posture.")
    return image

# Pose detection on an image
def detect_pose_on_image(image_path):
    model = YOLO('yolov8n-pose.pt')
    img = cv2.imread(image_path)
    results = model(img)
    if len(results) > 0:
        result = results[0]
        annotated_img = result.plot()
        keypoints = result.keypoints
        annotated_img = visual_keypoints(annotated_img, keypoints)
        cv2.imshow('Pose Detection', annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        results.save('pose_detected_image.jpg')
    else:
        print("No pose detected.")

# Function to perform pose detection on a video stream
def detect_pose_on_video(input_path, output_path=None, model_type='yolov8n-pose.pt'):
    
    # Load the pretrained YOLOv8-pose model
    model = YOLO(model_type)

    # Determine if the input is an image or video based on the file extension
    is_video = input_path.split('.')[-1].lower() in ['mp4', 'avi', 'mov', 'mkv']

    # Process an image
    if not is_video:
        # Run inference on the image
        results = model(input_path)
        
        # Display the result
        results.show()

        # If an output path is provided, save the output image
        if output_path:
            output_image = results.render()[0]  # Get rendered image
            cv2.imwrite(output_path, output_image)
            print(f"Output saved to {output_path}")

    else:
        # Process a video
        cap = cv2.VideoCapture(input_path)

        # Get video properties for output video
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Define the codec and create VideoWriter object if output path is provided
        if output_path:
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        else:
            out = None

        # Loop through each frame of the video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference on the current frame
            results = model(frame)[0]  # Predict on an image
            boxes = results.boxes.xyxy.tolist()

            # Keypoints object for pose outputs
            keypoints = results.keypoints.xyn.tolist()
            image = results.plot(boxes=False, kpt_radius=2)
            # image = visual_keypoints(image,keypoints)
            image = check_posture(image, keypoints, box=None, postures_to_check=["back", "shoulder", "leg"])
            # Display the frame
            cv2.imshow('YOLOv8 Pose Estimation', image)

            # Write the frame to the output video if required
            if out:
                out.write(frame)

            # Press 'q' to stop early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release video objects
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

        if output_path:
            print(f"Output video saved to {output_path}")


    # Release the video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()

# Main function to handle image or video input
if __name__ == "__main__":
    # Example: Detect pose on a static image
    image_path = 'path/to/your/image.jpg'  # Provide your image path
    video_path="input_data/person.mp4"
    output_path="output_data"
    # detect_pose_on_image(image_path)
    detect_pose_on_video(input_path=video_path,output_path=output_path)

