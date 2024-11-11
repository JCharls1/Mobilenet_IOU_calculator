import os
import tensorflow as tf
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Load the TFLite model and allocate tensors
model_path = r'C:\Users\garci\Desktop\kodings\mobilenetv1\mobaylnet.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# Load pbtxt file for class mapping
def load_class_mapping(pbtxt_path):
    class_mapping = {}
    with open(pbtxt_path, 'r') as file:
        lines = file.readlines()
        current_id = None
        current_name = None
        for line in lines:
            line = line.strip()
            if line.startswith('id:'):
                current_id = int(line.split(':')[1].strip())
            elif line.startswith('display_name:'):
                current_name = line.split(':')[1].strip().strip('"')
            if current_id is not None and current_name is not None:
                class_mapping[current_id] = current_name
                current_id, current_name = None, None
    return class_mapping

# Preprocess the input image
def preprocess_image(image_path):
    image = Image.open(image_path).resize((input_shape[1], input_shape[2]))
    input_data = np.array(image).astype(np.uint8)
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

# Interpret the output from the model
def interpret_output(output_data, threshold=0.5):
    boxes = output_data[0][0]
    class_ids = output_data[1][0]
    scores = output_data[2][0]
    results = []
    for i in range(len(scores)):
        if scores[i] >= threshold:
            result = {
                "bounding_box": boxes[i],
                "class_id": int(class_ids[i]),
                "score": scores[i]
            }
            results.append(result)
    return results

# Function to calculate Intersection over Union (IoU)
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0

# Draw a bounding box manually with real-time visualization
def draw_manual_bbox(image_path):
    image = Image.open(image_path)
    fig, ax = plt.subplots()
    ax.imshow(image)
    bbox = []
    rect = Rectangle((0, 0), 0, 0, linewidth=2, edgecolor='blue', facecolor='none')
    ax.add_patch(rect)

    def on_press(event):
        if event.xdata and event.ydata:
            bbox.clear()
            bbox.append((event.xdata, event.ydata))
            rect.set_xy((event.xdata, event.ydata))
            rect.set_width(0)
            rect.set_height(0)
            fig.canvas.draw()

    def on_motion(event):
        if event.xdata and event.ydata and bbox:
            x0, y0 = bbox[0]
            x1, y1 = event.xdata, event.ydata
            rect.set_width(x1 - x0)
            rect.set_height(y1 - y0)
            fig.canvas.draw()

    def on_release(event):
        if event.xdata and event.ydata:
            bbox.append((event.xdata, event.ydata))
            plt.close()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    plt.show()

    if len(bbox) == 2:
        x1, y1 = bbox[0]
        x2, y2 = bbox[1]
        return [x1, y1, x2, y2]
    else:
        return None
# Count true positives and false negatives for a specific class
true_positives = 0
false_negatives = 0

# Draw bounding boxes, calculate IoU, and count TP and FN for a specific class
def draw_boxes_and_evaluate(image_path, detection_results, class_mapping, ground_truth_box, target_class_name, output_path=None, iou_threshold=0.5):
    global true_positives
    global false_negatives

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Get the class ID for the target class name
    target_class_id = None
    for class_id, class_name in class_mapping.items():
        if class_name == target_class_name:
            target_class_id = class_id
            break

    if target_class_id is None:
        print(f"Class '{target_class_name}' not found in the class mapping.")
        return

    # Draw the ground-truth bounding box in blue
    draw.rectangle([(ground_truth_box[0], ground_truth_box[1]), 
                    (ground_truth_box[2], ground_truth_box[3])], outline="blue", width=2)
    draw.text((ground_truth_box[0], ground_truth_box[1] - 10), "Ground Truth", fill="blue")

    # Assume the ground truth box is for the target class
    false_negatives += 1  # Start with 1 FN to reduce if matched

    # Iterate over detection results and only consider the target class
    for result in detection_results:
        box = result["bounding_box"]
        class_id = result["class_id"]
        score = result["score"]
        class_name = class_mapping.get(class_id, "Unknown")

        if class_id == target_class_id:
            # Convert detected box (normalized coordinates) to actual dimensions
            box = [
                box[1] * image.width, box[0] * image.height,
                box[3] * image.width, box[2] * image.height
            ]

            # Draw the detected bounding box in red
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=2)
            draw.text((box[0], box[1] - 10), f"{class_name}: {score:.2f}", fill="red")

            # Calculate IoU with the ground-truth box
            iou = calculate_iou(box, ground_truth_box)
            draw.text((box[0], box[1] - 25), f"IoU: {iou:.2f}", fill="red")

            if iou >= iou_threshold:
                true_positives += 1
                false_negatives -= 1  # Reduce FN count if matched

    # Display true positive and false negative counts
    print(f"Evaluating for class: {target_class_name}")
    print(f"True Positives: {true_positives}")
    print(f"False Negatives: {false_negatives}")

    # Save or display the image
    if output_path:
        image.save(output_path)
    else:
        image.show()

# Process all images in a folder for a specific class
def process_folder(folder_path, output_folder, target_class_name):
    pbtxt_path = r'C:\Users\garci\Desktop\kodings\mobilenetv1\mscoco_label_map.pbtxt'
    class_mapping = load_class_mapping(pbtxt_path)

    # Iterate over all images in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            print(f"Processing {image_path}...")

            # Draw the bounding box manually
            ground_truth_box = draw_manual_bbox(image_path)
            if ground_truth_box:
                # Load and preprocess the image
                input_data = preprocess_image(image_path)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

                # Retrieve and interpret the output
                output_data = [
                    interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))
                ]
                detection_results = interpret_output(output_data)

                # Create the output file path
                output_path = os.path.join(output_folder, filename.replace(".jpg", "_evaluated.jpg"))

                # Draw bounding boxes and evaluate only for the target class
                draw_boxes_and_evaluate(image_path, detection_results, class_mapping, ground_truth_box, target_class_name, output_path)
            else:
                print(f"Bounding box drawing was not completed for {image_path}.")

# Example usage
folder_path = r'C:\Users\garci\Desktop\kodings\mobilenetv1\images'
output_folder = r'C:\Users\garci\Desktop\kodings\mobilenetv1\output'
target_class_name = "apple"  # Change this to the desired class name
os.makedirs(output_folder, exist_ok=True)
process_folder(folder_path, output_folder, target_class_name)
