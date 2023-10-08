import numpy as np
import cv2


def get_boxes(img, box_width=400, box_height=60, debug=False):
    """
    Extracts boxed zones in the image

    Steps:
    - Preprocess image
    - Detect contours
        - Check for 4 edges and bounding width, heigh, width/height ratio
        - Check for bounding rect angle
            - Correct perspective of box if 0 < angle <= 15
            - Else just resize
        - Save box
    - Return extracted boxes in sequential order

    Note:
    - Attempted SIFT and ORD with template, but doesn't work as well
    """
    # Greyscale and binarize image
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(grey,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,51,9)

    # Detect contours in image
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # For all contours
    boxes = []
    for c in contours:
        # If contour has approximately 4 edges
        approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            x,y,w,h = cv2.boundingRect(c)

            # If contour has acceptable width, height, and width/height ratio
            if w/h < (box_width/box_height)/2 or w < 100 or h < 15: 
                continue
            
            angle = cv2.minAreaRect(c)[-1]

            # If detected angle is within bounds, warp image
            if 0 < angle <= 15:
                # Correct box's perspective
                src_pts = np.array(approx.reshape(4,2), dtype=np.float32)                                                 # Source (box) points
                dest_pts = np.array([[0, 0], [0, box_height], [box_width, box_height], [box_width, 0]], dtype=np.float32) # Destination points
                M = cv2.getPerspectiveTransform(src_pts, dest_pts)                                                        # Perspective transformation
                box = cv2.warpPerspective(img, M, (box_width, box_height))                                         # Transformed image

            # Else forgo warping and just resize
            else:
                box = cv2.resize(img[y:y+h, x:x+w], (box_width, box_height))

            # Display before and after manipulating image
            if debug:
                    display("zone", img[y:y+h, x:x+w])
                    display("transformed", box)

            # Save box 
            boxes = [box] + boxes # Stack because bottom-up
            
    return boxes


def display(text, img):
    """
    Displays input image
    """
    cv2.imshow(text, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example
    img = cv2.imread("docs/test/digital_1.jpg")
    boxes = get_boxes(img, debug=True)
    print(len(boxes), "boxes detected")
