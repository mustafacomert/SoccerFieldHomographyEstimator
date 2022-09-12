import cv2
import numpy as np
import math

circles = np.zeros((4, 2), int)
count = 0

# finding matrix using dlt,
# first we obtianed a 8x9 matrix, then we apply svd on that matrix
# use the vh are 2D unitary arrays to find homography matrix
def find_matrix(p1, p2):
    A = []
    for i in range(0, len(p1)):
        x, y = p1[i][0], p1[i][1]
        u, v = p2[i][0], p2[i][1]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    A = np.asarray(A)
    u, s, vh = np.linalg.svd(A)
    arr = vh[-1, :] / vh[-1, -1]
    h = arr.reshape(3, 3)
    return h

# slightly bad version of the above function
# adds one additional row to the 8x9 matrix
# benefits of homogenous coordinates property h33 = 1
# makes it 9x9 thus 9x9 matrix is invertable

def find_matrix2(p1, p2):
    A = []
    for i in range(0, len(p1)):
        x, y = p1[i][0], p1[i][1]
        u, v = p2[i][0], p2[i][1]
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A.append([0, 0, 0, 0, 0, 0, 0, 0, 1])
    B = np.zeros((9, 1), int)
    B[8] = 1
    h = np.matmul(np.linalg.inv(A), B)
    return h.reshape(3, 3)


def convert_to_homogenous(pts):
    ret = np.zeros((4, 3), int)
    for i in range(len(pts)):
        ret[i] = circles[i][0], circles[i][1], 1
    return  ret

def convert_to_nonhomogenous(pts):
    ret = []
    if pts[2] != 0:
        ret.append(pts[0]/pts[2])
        ret.append(pts[1]/pts[2])
    else:
        return math.inf
    return ret


#takes two 3D homogeneous coordinates, returns cross product of them
def cross_product(pts1, pts2):
    ret = []
    i = pts1[1] * pts2[2] - pts1[2] * pts2[1]
    ret.append(i)
    j = pts1[0] * pts2[2] - pts1[2] * pts2[0]
    ret.append(j * -1)
    k = pts1[0] * pts2[1] - pts1[1] * pts2[0]
    ret.append(k)
    return ret


def find_intersection_point(pts1):
    # first i need to find two line equations
    hp = convert_to_homogenous(pts1)
    line1 = cross_product(hp[1], hp[3])
    line2 = cross_product(hp[0], hp[2])
    intersect = cross_product(line1, line2)
    return intersect


def mouse_points(event, x, y, flags, params):
    global count
    if count < 4 and event == cv2.EVENT_LBUTTONDOWN:
        circles[count] = x, y
        count = count + 1


def resize_wrt_min(im_list):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=cv2.INTER_CUBIC)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


imgSource = cv2.imread('resources/IMG_20200316_171745.jpg')
cv2.namedWindow('SourceImage', cv2.WINDOW_NORMAL)

imgTarget = cv2.imread('resources/soccer fieild.jpg')


while True:
    if count == 4:
        width, height = imgTarget.shape[1], imgTarget.shape[0]
        # intersection = find_intersection_point(circles)
        # fourth_point = convert_to_nonhomogenous(intersection)
        pts1 = np.float32([circles[0], circles[1], circles[2], circles[3]])
        # inter = find_intersection_point([[0, 0], [0, height], [width, 0], [width, height]])
        # asd = convert_to_nonhomogenous(inter)
        pts2 = np.float32([[0, 0], [0, height], [width, 0], [width, height]])
        homography_matrix1 = cv2.getPerspectiveTransform(pts1, pts2)
        homography_matrix2 = find_matrix(pts1, pts2)
        homography_matrix3 = find_matrix2(pts1, pts2)
        print("matrix1", homography_matrix1)
        print("matrix2", homography_matrix2)
        print("matrix3", homography_matrix3)
        imgOutput1 = cv2.warpPerspective(imgSource, homography_matrix1, (width, height))
        imgOutput2 = cv2.warpPerspective(imgSource, homography_matrix2, (width, height))
        imgOutput3 = cv2.warpPerspective(imgSource, homography_matrix3, (width, height))

        cv2.namedWindow('OutputImage1', cv2.WINDOW_NORMAL)
        imgOutput1 = resize_wrt_min([imgOutput1, imgTarget])
        cv2.imshow("OutputImage1", imgOutput1)

        cv2.namedWindow('OutputImage2', cv2.WINDOW_NORMAL)
        imgOutput2 = resize_wrt_min([imgOutput2, imgTarget])
        cv2.imshow("OutputImage2", imgOutput2)

        cv2.namedWindow('OutputImage3', cv2.WINDOW_NORMAL)
        imgOutput3 = resize_wrt_min([imgOutput3, imgTarget])
        cv2.imshow("OutputImage3", imgOutput3)
    for x in range(0, 4):
        cv2.circle(imgSource, (circles[x][0], circles[x][1]), 22, (0, 255, 0), cv2.FILLED)

    cv2.imshow("SourceImage", imgSource)
    cv2.setMouseCallback("SourceImage", mouse_points)
    cv2.waitKey(1)

