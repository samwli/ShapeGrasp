import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import pickle
from code.load_data import load_mask

# Load your image
# obj = 'soldering_iron'
# iter = '1'
# idx = 0
# image_path = f'data/{obj}{iter}/{obj}_rgb.png'
# image = cv2.imread(image_path)
# mask = load_mask(f'data/{obj}{iter}/{obj}_mask')
# mask = mask.astype(bool)
# # Apply the mask to the image
# image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8)*255)
# image[mask == 0] = [255, 255, 255]
# # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# mode = '2d'
# # iter = '?'
# hulls_path = f'outputs/{obj}{iter}_{mode}_object/{obj}_2d_hulls.pkl'

# with open(hulls_path, 'rb') as f:
#     hulls = pickle.load(f)
    
def draw_star(image, center, color, size=20):
    # Define the vertices for a 5-pointed star
    def star_points(center, size):
        points = []
        for i in range(5):
            outer_x = center[0] + int(size * np.cos(2 * np.pi * i / 5 - np.pi / 2))
            outer_y = center[1] + int(size * np.sin(2 * np.pi * i / 5 - np.pi / 2))
            points.append((outer_x, outer_y))

            inner_x = center[0] + int((size / 2) * np.cos(2 * np.pi * i / 5 + np.pi / 5 - np.pi / 2))
            inner_y = center[1] + int((size / 2) * np.sin(2 * np.pi * i / 5 + np.pi / 5 - np.pi / 2))
            points.append((inner_x, inner_y))
        return np.array([points], dtype=np.int32)

    # Get the points for the star
    star = star_points(center, size)

    # Draw the star
    cv2.fillPoly(image, star, color)
    
def areaPoly(points):
    area = 0
    nPoints = len(points)
    j = nPoints - 1
    i = 0
    for point in points:
        p1 = points[i]
        p2 = points[j]
        area += (p1[0]*p2[1])
        area -= (p1[1]*p2[0])
        j = i
        i += 1

    area /= 2
    return area
    
def centroidPoly(points):
    nPoints = len(points)
    x = 0
    y = 0
    j = nPoints - 1
    i = 0

    for point in points:
        p1 = points[i]
        p2 = points[j]
        f = p1[0]*p2[1] - p2[0]*p1[1]
        x += (p1[0] + p2[0])*f
        y += (p1[1] + p2[1])*f
        j = i
        i += 1

    area = areaPoly(points)
    f = area*6
    return [x/f, y/f]

def check_collision(new_rect, existing_rects):
    """Check if the new rectangle collides with any existing rectangle."""
    new_bottom_left, new_top_right = new_rect
    for existing_rect in existing_rects:
        existing_bottom_left, existing_top_right = existing_rect[0], existing_rect[1]
        # If one rectangle is on left side of other
        if new_top_right[0] < existing_bottom_left[0] or existing_top_right[0] < new_bottom_left[0]:
            continue
        # If one rectangle is above other
        if new_bottom_left[1] > existing_top_right[1] or existing_bottom_left[1] > new_top_right[1]:
            continue
        return True  # Rectangles overlap
    return False  # No collision
    
# Function to draw convex hulls and label them
def draw_convex_hulls(image, list_of_hulls):
    color_bank = [
        (255, 0, 0),      # Blue
        (0, 255, 0),      # Green
        (0, 0, 255),      # Red
        (0, 255, 255),    # Yellow
        (255, 0, 255),    # Magenta
        (255, 255, 0),    # Cyan
        (0, 127, 255),    # Orange
        (255, 0, 127),    # Rose
        (127, 255, 0),    # Chartreuse
        (255, 127, 0),    # Amber
        (0, 255, 127),    # Spring Green
        (127, 0, 255),    # Electric Violet
        (255, 127, 127),  # Salmon Pink
        (127, 255, 127),  # Mint Green
        (127, 127, 255),  # Lavender
    ]

    used_colors = []
    drawn_rectangles = []
    centers = []
    for index, hull in enumerate(list_of_hulls):
        if hull.volume < 500:
            continue
        # Generate a unique color for each hull with a random RGB and fixed alpha for transparency
        color_bgr = color_bank[index % len(color_bank)]
        if color_bgr not in used_colors or len(used_colors) == len(color_bank):
            used_colors.append(color_bgr)
        if len(used_colors) == len(color_bank):  # Reset used colors if all have been used
            used_colors = []

        is_color_dark = (0.299 * color_bgr[2] + 0.587 * color_bgr[1] + 0.114 * color_bgr[0]) < 128
        text_color = (255, 255, 255) if is_color_dark else color_bgr
        alpha = 0.2
        
        # Draw the convex hull lines
        for simplex in hull.simplices:
            start_point = tuple(map(int, hull.points[simplex[0]]))  # Convert to tuple of ints
            end_point = tuple(map(int, hull.points[simplex[1]]))    # Convert to tuple of ints
            cv2.line(image, start_point, end_point, color_bgr, 2)
            
        # Create a mask where we will draw the hull
        mask = np.zeros_like(image)
        
        # Fill the convex hull with white color on the mask
        cv2.fillConvexPoly(mask, np.array(hull.points[hull.vertices], dtype=np.int32), (255, 255, 255))
        
        # Alpha blend the mask with the image
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        image_hull = cv2.bitwise_and(image, image, mask=mask)
        color_image = np.full_like(image_hull, color_bgr)
        cv2.addWeighted(color_image, alpha, image_hull, 1 - alpha, 0, image_hull)
        
        # Copy the blended hull back into the image
        image[mask > 0] = image_hull[mask > 0]
        # Get the center for the label
        hull_points = hull.points[hull.vertices]
        cx, cy = centroidPoly(hull_points)
        center = (int(cx), int(cy))

        # Draw a solid black rectangle for the index background
        rectangle_bgr = (0, 0, 0)
        rectangle_size = (30, 20)  # Width, Height of the rectangle
        bottom_left_corner = (center[0] - rectangle_size[0] // 2, center[1] - rectangle_size[1] // 2)
        top_right_corner = (center[0] + rectangle_size[0] // 2, center[1] + rectangle_size[1] // 2)
        # while check_collision((bottom_left_corner, top_right_corner), drawn_rectangles):
        #     center = (center[0], center[1] + 10)  # Move down by 10 pixels
        #     bottom_left_corner = (center[0] - rectangle_size[0] // 2, center[1] - rectangle_size[1] // 2)
        #     top_right_corner = (center[0] + rectangle_size[0] // 2, center[1] + rectangle_size[1] // 2)

        # Add the (possibly adjusted) rectangle to the list of drawn rectangles
        drawn_rectangles.append((bottom_left_corner, top_right_corner))
        centers.append(center)
        cv2.rectangle(image, bottom_left_corner, top_right_corner, rectangle_bgr, -1)

        # Put the index number on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(str(index), font, font_scale, font_thickness)[0]
        text_x = center[0] - text_size[0] // 2
        text_y = center[1] + text_size[1] // 2
        cv2.putText(image, str(index), (text_x, text_y), font, font_scale, text_color, font_thickness)
        # yellow_bgr = (0, 255, 255)
        # Draw a small yellow solid star at the center
        # draw_star(image, center, yellow_bgr, size=15)

        # You can still add the center to the list of drawn rectangles if needed for collision checking
        # Just for demonstration, adjust according to your collision detection needs
        drawn_rectangles.append((center[0] - 10, center[1] - 10, center[0] + 10, center[1] + 10))
    return image, centers

# Draw the convex hulls on the image
# image_with_hulls, centers = draw_convex_hulls(image, hulls)


# # plt.axis('off')  # Hide the axis
# cv2.imwrite(f'{obj}_affordance.png', image_with_hulls)
