import cv2
def function(inputs,options):
		image = inputs["image"]
		left, top, right, bottom = inputs["boxes"]
		image[top:bottom,left:right] = cv2.blur(image[top:bottom,left:right],(30,30))
		return {"image":image}