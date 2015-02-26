import cv2;

def online_variance(new_data,curr_var,curr_iter,curr_mean):
	if curr_iter==1:
		new_mean = new_data;
		new_var = 0;
		return new_mean,new_var;
	else:

		pa=cv2.subtract(new_data,curr_mean);
		pa=cv2.divide(pa,curr_iter,1);
		new_mean=cv2.add(pa,curr_mean);
		#new_mean = curr_mean + (new_data - curr_mean)/curr_iter;
	
		prev_S = curr_var * (curr_iter - 2);
	
		#
		pd1=cv2.subtract(new_data,curr_mean);
		pd2=cv2.subtract(new_data,new_mean);
		pd=cv2.multiply(pd1,pd2);
		new_S=cv2.add(pd,prev_S);
		#new_S = prev_S  + (new_data  - curr_mean) .* (new_data - new_mean);
		
		new_var=cv2.divide(new_S,curr_iter-1);
		#new_var = new_S/(curr_iter - 1);
		
		return new_mean,new_var;


img = cv2.imread('OrigImg0002.png',0);
netimg1,netimg2=online_variance(img,1,2,1);
cv2.imshow('img',netimg1)
cv2.waitKey(0)
cv2.destroyAllWindows()




