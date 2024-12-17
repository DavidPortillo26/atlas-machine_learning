import numpy as np

def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
    """
    Suppresses non-maximal bounding boxes.

    Parameters:
        filtered_boxes (numpy.ndarray): Array of filtered bounding boxes.
        box_classes (numpy.ndarray): Class indices for each bounding box.
        box_scores (numpy.ndarray): Confidence scores for each bounding box.

    Returns:
        tuple: (box_predictions, predicted_box_classes, predicted_box_scores)
    """
    box_predictions = []
    predicted_box_classes = []
    predicted_box_scores = []

    # Get the unique class indices
    unique_classes = np.unique(box_classes)

    for cls in unique_classes:
        # Get boxes and scores for the current class
        indices = np.where(box_classes == cls)
        cls_boxes = filtered_boxes[indices]
        cls_scores = box_scores[indices]

        # Sort by score in descending order
        sorted_indices = np.argsort(-cls_scores)
        cls_boxes = cls_boxes[sorted_indices]
        cls_scores = cls_scores[sorted_indices]

        # Perform Non-Max Suppression
        while len(cls_boxes) > 0:
            # Add the highest scoring box
            box_predictions.append(cls_boxes[0])
            predicted_box_classes.append(cls)
            predicted_box_scores.append(cls_scores[0])

            # Compute IoU for remaining boxes
            ious = self._iou(cls_boxes[0], cls_boxes[1:])

            # Filter boxes with IoU < nms_t
            remaining_indices = np.where(ious < self.nms_t)
            cls_boxes = cls_boxes[1:][remaining_indices]
            cls_scores = cls_scores[1:][remaining_indices]

    # Convert lists to numpy arrays
    box_predictions = np.array(box_predictions)
    predicted_box_classes = np.array(predicted_box_classes)
    predicted_box_scores = np.array(predicted_box_scores)

    return box_predictions, predicted_box_classes, predicted_box_scores

def _iou(self, box1, boxes):
    """
    Calculates Intersection over Union (IoU) between a box and multiple boxes.

    Parameters:
        box1 (numpy.ndarray): Single bounding box [x1, y1, x2, y2].
        boxes (numpy.ndarray): Array of bounding boxes [x1, y1, x2, y2].

    Returns:
        numpy.ndarray: IoU values for each box.
    """
    # Calculate intersection
    x1 = np.maximum(box1[0], boxes[:, 0])
    y1 = np.maximum(box1[1], boxes[:, 1])
    x2 = np.minimum(box1[2], boxes[:, 2])
    y2 = np.minimum(box1[3], boxes[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Calculate area of both boxes
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Calculate union
    union = area_box1 + area_boxes - intersection

    # Avoid division by zero
    iou = intersection / np.maximum(union, 1e-9)

    return iou
