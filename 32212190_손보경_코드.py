import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist

#line

def region_ofinterest(img, vertices, color3 = (255,255,255), color1 = 255) :
    mask = np.zeros_like(img)
    
    if len(img.shape) > 2:
        color = color3
    else :
        color = color1
        
    cv2.fillPoly(mask, vertices, color)
    cv2.imshow('mask', mask)
    ROI_image = cv2.bitwise_and(img, mask)
    
    return ROI_image



def draw_lines(img, lines, color = [0, 0, 255], thickness = 15) :
    for line in lines :
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap) :
    lines = cv2.HoughLinesP(img, rho, theta, threshold, minLineLength = min_line_len, maxLineGap = max_line_gap)
    
    return lines



def weighted_img(img, img2, a=1, b=1., c=0.) :
    return cv2.addWeighted(img2, a, img, b, c)

def avg_lines(lines) :
    avg_fit = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
	    
            avg_fit.append(slope)
	
        avg_line = np.average(avg_fit)

        return avg_line


class ImageSub():
    def __init__(self) :
        # subscribes raw image
        self.sub_image_original = rospy.Subscriber('/camera/image', Image, self.cbImage, queue_size =1)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size = 1)
        self.twist = Twist()
        self.twist.angular.z = 0.03
        self.cvBridge = CvBridge()
	

    def cbImage(self, msg_img) :
        cv_image_original = self.cvBridge.imgmsg_to_cv2(msg_img, "bgr8")
        (rows, cols, channels) = cv_image_original.shape

        img_w = cv2.inRange(cv_image_original, (220,220,220), (255, 255, 255))
        img = cv2.cvtColor(cv_image_original, cv2.COLOR_BGR2HSV)
        img_y = cv2.inRange(img, (10,100,100), (80, 255, 255))
	
        img = weighted_img(img_w, img_y)
        
        canny_img = cv2.Canny(img, 200, 400)
	
        height, width = cv_image_original.shape[:2]

        vertices = np.array([[(10, height), (20, height*0.85), (width-20, height*0.85), (width-10, height)]], dtype = np.int32)

        ROI_img = region_ofinterest(canny_img, vertices)

	
        line_arr = hough_lines(ROI_img, 1, 1*np.pi/180, 5, 5, 30)
        line_arr = np.squeeze(line_arr)

	
        # 기울기 구하기
        slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi

        # 수평 기울기 제한
        line_arr = line_arr[np.abs(slope_degree) < 160]
        slope_degree = slope_degree[np.abs(slope_degree) < 160]

        # 수직 기울기 제한
        line_arr = line_arr[np.abs(slope_degree) > 100]
        slope_degree = slope_degree[np.abs(slope_degree) > 100]

        # 필터링된 직선 버리기
        L_lines, R_lines = line_arr[(slope_degree > 0), :], line_arr[(slope_degree < 0), :]
        temp = np.zeros((cv_image_original.shape[0], cv_image_original.shape[1], 3), dtype = np.uint8)
        L_lines, R_lines = L_lines[:, None], R_lines[:, None]
	

        # 직선 그리기
        draw_lines(temp, L_lines)
        draw_lines(temp, R_lines)


        result = weighted_img(temp, cv_image_original)

        cv2.imshow('Result',result)

        avg_slope_left = np.abs(avg_lines(L_lines))
        avg_slope_right = np.abs(avg_lines(R_lines))
	
	
        if np.isnan(avg_slope_left) :
            self.twist.linear.x = 0.39
            self.twist.angular.z = (3 - avg_slope_right)*0.5
        elif np.isnan(avg_slope_right) :
            self.twist.linear.x = 0.39
            self.twist.angular.z = -(3 - avg_slope_left)*0.5
        else :
            self.twist.linear.x = 0.48
	
	
        self.cmd_vel_pub.publish(self.twist)
	
        cv2.waitKey(3)
	

    def main(self) :
        rospy.spin()


if __name__ == '__main__' :
    rospy.init_node('image_sub')
    node = ImageSub()
    node.main()
