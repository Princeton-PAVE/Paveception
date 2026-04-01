import cv2
import numpy as np


# Load your specific images
img1 = cv2.imread("./photos/img1b.jpg")
img2 = cv2.imread("./photos/img1a.jpg")
print(img1)
print(img2)

# Convert images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Find keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Use FLANN-based matcher to find matches between descriptors
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

raw_matches = flann.knnMatch(des1, des2, k=2)

# Lowe's ratio test to filter good matches
good_matches = []
for m, n in raw_matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

if len(good_matches) < 4:
    raise RuntimeError(f"Not enough good matches found: {len(good_matches)}")

# Draw top 10 matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 4. Find Homography
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 5. Warp img1 to img2's plane
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

# Compute transformed corners of img1 and determine canvas size
pts_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
pts_img1_trans = cv2.perspectiveTransform(pts_img1, H)
pts_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

all_pts = np.concatenate((pts_img1_trans, pts_img2), axis=0)
xmin, ymin = np.int32(all_pts.min(axis=0).ravel() - 0.5)
xmax, ymax = np.int32(all_pts.max(axis=0).ravel() + 0.5)

translation = [-xmin, -ymin]
canvas_w = xmax - xmin
canvas_h = ymax - ymin

# Warp img1 into the canvas with translation to keep everything positive
translate_mat = np.array([[1, 0, translation[0]],
                          [0, 1, translation[1]],
                          [0, 0, 1]])
result = cv2.warpPerspective(img1, translate_mat.dot(H), (canvas_w, canvas_h))

# Paste img2 into canvas
canvas_img2 = np.zeros_like(result)
canvas_img2[translation[1]:translation[1]+h2, translation[0]:translation[0]+w2] = img2

mask1 = (result.sum(axis=2) > 0).astype(np.float32)
mask2 = (canvas_img2.sum(axis=2) > 0).astype(np.float32)

mask1 = cv2.GaussianBlur(mask1, (31, 31), 0)
mask2 = cv2.GaussianBlur(mask2, (31, 31), 0)

total = mask1 + mask2 + 1e-6
w1 = mask1 / total
w2 = mask2 / total

result = (result * w1[..., None] + canvas_img2 * w2[..., None]).astype(np.uint8)

# Show and save
# cv2.imshow('Stitched Result', result)
cv2.imwrite('stitched_output.png', result)
print("Success! Saved as 'stitched_output.png'")






cv2.waitKey(0)
cv2.destroyAllWindows()

