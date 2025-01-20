import numpy as np
import matplotlib.pyplot as plt
import cv2
import networkx as nx
import pickle
import os
from scipy.spatial import distance
from scipy import stats
from collections import Counter
from code.load_data import load_height, load_mask

def point_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def find_apex(points):
    side_lengths = [point_distance(points[i], points[(i+1)%3]) for i in range(3)]
    side_differences = [abs(side_lengths[i] - (side_lengths[(i+1)%3] + side_lengths[(i+2)%3]) / 2) for i in range(3)]
    apex_index = np.argmax(side_differences)-1
    
    return apex_index, side_lengths

def extend_base(apex, base_point, target_length):
    apex = apex.astype(float)
    base_point = base_point.astype(float)
    direction_vector = base_point - apex
    direction_vector /= np.linalg.norm(direction_vector)
    new_base_point = apex + direction_vector * target_length
    
    return new_base_point

def get_triangle(points):
    apex_index, side_lengths = find_apex(points)
    apex = points[apex_index]
    base_points = np.delete(points, apex_index, axis=0)
    point_distances_to_apex = [point_distance(apex, base_point) for base_point in base_points]
    farther_base_index = np.argmax(point_distances_to_apex)
    farther_base_point = base_points[farther_base_index]
    closer_base_point = base_points[1 - farther_base_index]
    new_base_point = extend_base(apex, closer_base_point, point_distances_to_apex[farther_base_index])
    
    return np.array([farther_base_point, new_base_point, apex])

def calculate_triangle_angle(base_point1, base_point2, apex_point):
    midpoint = (np.array(base_point1) + np.array(base_point2)) / 2
    apex_vector = np.array(apex_point) - midpoint
    angle_radians = np.arctan2(apex_vector[1], apex_vector[0])
    angle_degrees = np.degrees(angle_radians)
    if point_distance(base_point1, base_point2) < point_distance(base_point1, apex_point):
        return (angle_degrees+90) % 360 + 90
    else:
        return (angle_degrees+90) % 360

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

def circularity_ratio(hull):
    perimeter = cv2.arcLength(hull, True)
    area = cv2.contourArea(hull)
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return circularity

def is_close_to_circle(hull, threshold):
    circularity = circularity_ratio(hull)
    return circularity > threshold

def approximate_shape(hull, convex_hull):
    if is_close_to_circle(hull, 0.9):
        (x, y), radius = cv2.minEnclosingCircle(hull)
        return "circle", ((x, y), radius)
    
    epsilon = 0.015 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    if len(approx) == 1:
        return "circle", (approx.squeeze(), 1)
    elif len(approx) == 2:
        return "thin_ellipse", create_custom_ellipse(approx)
    elif len(approx) == 3:
        return "isosceles triangle", get_triangle(approx.squeeze())
    elif (len(approx) > 3 and len(approx) < 7):
        rect = cv2.minAreaRect(hull)
        return "rectangle", rect
    else:
        ellipse = cv2.fitEllipse(hull)
        return "ellipse", ellipse

def draw_shapes_on_image(img, hulls, output_dir, object_name):
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for hull in hulls:
        hull_vs = np.array(hull.points[hull.vertices]).astype(np.int32)
        if hull.volume < 500:
            continue
        epsilon = 0.015 * cv2.arcLength(hull_vs, True)
        approx = cv2.approxPolyDP(hull_vs, epsilon, True)
        if len(approx) < 3:
            continue
        shape, approx = approximate_shape(hull_vs, hull)
        if shape == "circle":
            center, radius = approx
            center = tuple(map(int, center))
            cv2.circle(img, center, int(radius), (255, 255, 0), 2)
        elif shape == "ellipse":
            cv2.ellipse(img, approx, (255, 255, 0), 2)
        elif shape == "rectangle":
            box = cv2.boxPoints(approx)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (255, 255, 0), 2)
        elif shape == 'isosceles triangle':  # isosceles triangle
            points = [tuple(pt) for pt in approx.squeeze()]
            cv2.polylines(img, [np.array(points, np.int32)], True, (255, 255, 0), 2)
    cv2.imwrite(output_dir+f'/{object_name}_shapes.png', img)   
    return img

def point_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


def shared_boundaries(hull1, hull2):
    for s1 in hull1.simplices:
        for s2 in hull2.simplices:
            start1, end1 = hull1.points[s1[0]], hull1.points[s1[1]]
            start2, end2 = hull2.points[s2[0]], hull2.points[s2[1]]
            if (point_distance(start1, start2) < 1 and point_distance(end1, end2) < 1) or (point_distance(start1, end2) < 1 and point_distance(end1, start2) < 1):
                return (point_distance(start1, end1), (start1[0], start1[1]), (end1[0], end1[1]))
            if (point_distance(start1, start2) < 1 or point_distance(end1, end2) < 1) or (point_distance(start1, end2) < 1 or point_distance(end1, start2) < 1):
                return (1, (start1[0], start1[1]), (end1[0], end1[1]))
                        
    return None
    
def find_node_name(nodes, number):
    number_str = str(number)
    for node in nodes:
        if node.endswith(number_str):
            return node
    return None

def get_height(hull, height_img, binary_mask):
    height_img = 100*height_img/height_img.max()
    mask = np.zeros(height_img.shape[:2], dtype=np.uint8)
    for simplex in hull.simplices:
        pt1 = tuple(hull.points[simplex[0]].astype(int))
        pt2 = tuple(hull.points[simplex[1]].astype(int))
        cv2.line(mask, pt1, pt2, 255, 1)
    hull_points = np.array([hull.points[vertex].astype(int) for vertex in hull.vertices])
    cv2.fillPoly(mask, [hull_points], 1)
    # TODO: adjust height threshold for real world
    condition = (mask == 1)
    selected_depth_values = height_img[condition]
    average_depth = np.mean(selected_depth_values)
    
    return average_depth

def closest_color(pixel, web_colors_np):
    min_distance = float('inf')
    closest_color_name = None
    for color_name, rgb_values in web_colors_np.items():
        for rgb in rgb_values:
            dist = distance.euclidean(pixel, rgb)
            if dist < min_distance:
                min_distance = dist
                closest_color_name = color_name
    return closest_color_name

def bucket_to_web_colors(image, web_colors):
    # Adjust for list of RGB values
    web_colors_np = {k: [np.array(v) for v in values] for k, values in web_colors.items()}
    bucketed_list = []
    for i in range(image.shape[0]):
        closest = closest_color(image[i], web_colors_np)
        bucketed_list.append(closest)
    
    return bucketed_list

def find_most_common_color(image):
    web_colors = {
        "White": [(225, 225, 225), (255, 255, 255)],
        "Silver": [(192, 192, 192)],
        "Gray": [(62, 62, 62), (128, 128, 128), (158, 158, 158)],
        "Brown": [(90, 66, 50)],
        "Beige": [(210, 205, 180), (190, 170, 130)],
        "Black": [(0, 0, 0), (32, 32, 32)],
        "Red": [(255, 0, 0), (132, 20, 10)],
        "Orange": [(150, 75, 25)],
        "Maroon": [(96, 0, 0)],
        "Yellow": [(255, 255, 0)],
        "Olive": [(128, 128, 0)],
        "Green": [(0, 128, 0), (70, 110, 40)],
        "Aqua": [(0, 255, 255)],
        "Teal": [(0, 152, 152)],
        "Blue": [(0, 0, 255)],
        "Navy": [(48, 48, 128), (0, 0, 60)],
        "Fuchsia": [(255, 0, 255)],
        "Purple": [(52, 16, 52)]
    }
    bucketed_list = bucket_to_web_colors(image, web_colors)
    counts = Counter(bucketed_list)
    most_common = counts.most_common(1)
    return most_common[0][0] if most_common else None

def get_color(hull, rgb_img, binary_mask):
    hull_mask = np.zeros(rgb_img.shape[:2], dtype=np.uint8)
    for simplex in hull.simplices:
        pt1 = tuple(hull.points[simplex[0]].astype(int))
        pt2 = tuple(hull.points[simplex[1]].astype(int))
        cv2.line(hull_mask, pt1, pt2, 255, 1)
    hull_points = np.array([hull.points[vertex].astype(int) for vertex in hull.vertices])
    cv2.fillPoly(hull_mask, [hull_points], 1)
    condition = (hull_mask == 1) & (binary_mask == 1)
    selected_rgb_values = rgb_img[condition]
    rows_as_tuples = [tuple(row) for row in selected_rgb_values]
    color = find_most_common_color(selected_rgb_values)
    return color

def create_custom_ellipse(points):
    """
    Calculate the minimum area rectangle for a given set of points.
    - For a single point, it creates a 3x3 square centered at that point.
    - For two points, it creates a rectangle with length equal to the distance 
      between the points and width 3, perpendicular to the line connecting the points.
    """
    # Handle two points
    point1, point2 = np.squeeze(np.array(points, dtype=np.float32))
    center = tuple(((point1 + point2) / 2).astype(int))
    axis_length = int(np.linalg.norm(point1 - point2) / 2), 1  # Semi-major axis, semi-minor axis

    # Calculate angle of rotation for the ellipse
    dx, dy = point2 - point1
    angle = np.arctan2(dy, dx) * (180 / np.pi)  # Convert radians to degrees

    return (center, axis_length, angle)


def create_graph(hulls, output_dir, obj_data_path, mode):
    # with open(os.path.join(output_dir, f'{object_name}_2d_hulls.pkl'), 'rb') as f:
    #     hulls = pickle.load(f)
    object_name = output_dir.split('/')[1].split('_'+mode)[0][:-1]
    if mode == '3d':
        height_img = load_height(obj_data_path+'_depth')

    if os.path.exists(obj_data_path+'_rgb.npy'):
        rgb_img = np.load(obj_data_path+'_rgb.npy')
    elif os.path.exists(obj_data_path+'_rgb.png'):
        rgb_img = cv2.imread(obj_data_path+'_rgb.png')
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    else:
        raise FileNotFoundError("Neither NPY nor PNG file found for the given path")
    
    binary_mask = load_mask(obj_data_path+'_mask')
    # Apply the mask to the image
    rgb_img = cv2.bitwise_and(rgb_img, rgb_img, mask=binary_mask.astype(np.uint8)*255)
    cv2.imwrite(output_dir+f'/{object_name}_masked_rgb.png', cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
    G = nx.Graph()
    for idx, hull in enumerate(hulls):
        if hull.volume < 500:
            continue
        hull_vs = np.array(hull.points[hull.vertices]).astype(np.int32)
        epsilon = 0.015 * cv2.arcLength(hull_vs, True)
        approx = cv2.approxPolyDP(hull_vs, epsilon, True)
        if len(approx) < 3:
            continue
        shape, obj = approximate_shape(hull_vs, hull)
        hull_points = hull.points[hull.vertices]
        cx, cy = centroidPoly(hull_points)
        area = str(int(hull.volume / 400)) + ' cm^2'
        if shape == 'isosceles triangle':
            angle = calculate_triangle_angle(obj[0], obj[1], obj[2])
            base = point_distance(obj[0], obj[1])
            midpoint = [(obj[0][0]+obj[1][0])/2, (obj[0][1]+obj[1][1])/2]
            altitude = point_distance(midpoint, obj[2])
            aspect_ratio = max(base, altitude) / min(base, altitude)
            color = get_color(hull, rgb_img, binary_mask)
            G.add_node('tri{}'.format(idx), shape=shape, centroid=(int(cx), int(cy)), area=area, aspect_ratio = np.round(aspect_ratio, 3), angle = int(angle), color=color)
        elif shape == 'rectangle':
            width, height = obj[1]
            angle = obj[2]
            if width < height:
                angle = (angle + 90) % 180
                aspect_ratio = height / width if width != 0 else 0
            else:
                angle = (angle) % 180
                aspect_ratio = width / height if height != 0 else 0
                width = height
            width = str(int(width / 20)) + ' cm'
            color = get_color(hull, rgb_img, binary_mask)
            G.add_node('rect{}'.format(idx), shape=shape, centroid=(int(cx), int(cy)), area=area, aspect_ratio=np.round(aspect_ratio, 3), width=width, angle=int(angle), color=color)
        elif shape == 'ellipse':
            center, axes, angle = obj
            angle = (angle-90) % 180
            aspect_ratio = axes[1] / axes[0] if axes[0] != 0 else 0
            color = get_color(hull, rgb_img, binary_mask)
            G.add_node('ellip{}'.format(idx), shape=shape, centroid=(int(cx), int(cy)), area=area, aspect_ratio = np.round(aspect_ratio, 3), angle = int(angle), color=color)
        else:
            color = get_color(hull, rgb_img, binary_mask)
            G.add_node('circ{}'.format(idx), shape=shape, centroid=(int(cx), int(cy)), area=area, aspect_ratio = int(1), angle = int(0), color=color)
                
    node_names = list(G.nodes)
    for i in range(len(hulls)):
        if hulls[i].volume < 500:
            continue
        hull_vs = np.array(hulls[i].points[hulls[i].vertices]).astype(np.int32)
        epsilon = 0.015 * cv2.arcLength(hull_vs, True)
        approx = cv2.approxPolyDP(hull_vs, epsilon, True)
        if len(approx) < 3:
            continue
        for j in range(i+1, len(hulls)):
            if hulls[j].volume < 500:
                continue
            hull_vs = np.array(hulls[j].points[hulls[j].vertices]).astype(np.int32)
            epsilon = 0.015 * cv2.arcLength(hull_vs, True)
            approx = cv2.approxPolyDP(hull_vs, epsilon, True)
            if len(approx) < 3:
                continue
            boundary_info = shared_boundaries(hulls[i], hulls[j])
            if boundary_info:
                length, start_vertex, end_vertex = boundary_info
                # s1, s2 = int(start_vertex[0]), int(start_vertex[1])
                # e1, e2 = int(end_vertex[0]), int(end_vertex[1])
                G.add_edge(find_node_name(node_names, i), find_node_name(node_names, j), length=int(length)) #, start_vertex=(s1, s2), end_vertex=(e1, e2))
                
    shapes = draw_shapes_on_image(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR), hulls, output_dir, object_name)
    with open(output_dir + f'/{object_name}_graph.txt', 'w') as file:
        nodes = G.nodes(data=True)
        edges = G.edges(data=True)
        output_string = (
            "Nodes of the graph:\n" +
            str(nodes) +
            "\n\n" +
            "Edges of the graph:\n" +
            str(edges)
        )
        file.write(output_string)
    return output_string, shapes

    



    
