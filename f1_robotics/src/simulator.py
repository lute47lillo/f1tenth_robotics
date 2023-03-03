import numpy as np
    
    
# drives toward the furthest point it sees
class AnotherDriver:

    def process_lidar(self, ranges):
        # the number of LiDAR points
        NUM_RANGES = len(ranges)
        # angle between each LiDAR point
        ANGLE_BETWEEN = 2*np.pi / NUM_RANGES
        # number of points in each quadrant
        NUM_PER_QUADRANT = NUM_RANGES // 4

        # the index of the furthest LiDAR point (ignoring the points behind the car)
        max_idx = np.argmax(ranges[NUM_PER_QUADRANT:-NUM_PER_QUADRANT]) + NUM_PER_QUADRANT
        # some math to get the steering angle to correspond to the chosen LiDAR point
        steering_angle = max_idx*ANGLE_BETWEEN - (NUM_RANGES//2)*ANGLE_BETWEEN
        speed = 5.0
        
        return speed, steering_angle
    
class Driver:
    
    def create_bubble(self, closest, range):
        bubble = 100
        min = closest - bubble
        if min < 0:
            min = 0
            
        max = closest + bubble
        if max >= len(range):
            max = len(range)-1
        
        # Set points inside bubble to 0. Non-zero points are free space
        range[min: max] = 0
        
        return range
        
    def get_angle(self, range_index, range_len, angle_rad):
        lidar_angle = (range_index - (range_len/2)) * angle_rad
        steering_angle = lidar_angle / 2
        return steering_angle

    def process_lidar(self, ranges):
        
        # the number of LiDAR points
        n_ranges = len(ranges)
        
        # angle in radians between each LiDAR point
        angle_rad = (2*np.pi) / n_ranges
        
        # Divide LiDAR points in 4 spaces
        n_quad = n_ranges // 4
        
        # Get set of ranges of LiDAR points
        ranges_ahead = np.array(ranges[120:-120])
        ranges_ahead = np.convolve(ranges_ahead, np.ones(3), 'same') / 3
        ranges_ahead = np.clip(ranges_ahead, 0, 3000000)
        
        # Closest point
        closest_point = ranges_ahead.argmin()
        
        # Mask for 0s
        range_bubble = self.create_bubble(closest_point, ranges_ahead)
        masked = np.ma.masked_where(range_bubble==0, range_bubble)
        
        # Get longest non-zero sequence 
        slices = np.ma.notmasked_contiguous(masked)
        print(slices[0].start, slices[0].stop)
        
        start_gap = slices[0].start
        end_gap = slices[0].stop 

        
        averaged_max_gap = np.convolve(ranges[start_gap:end_gap], np.ones(150), 'same') / 150
        best_gap = averaged_max_gap.argmax() + start_gap
        
        # steering_angle = (best_gap - (len(range_bubble)/2)) * angle_rad
        # steering_angle = steering_angle / 2
        steering_angle = self.get_angle(best_gap, len(range_bubble), angle_rad)
        
        straight_angle = np.pi / 18
        print(straight_angle)
        if abs(steering_angle) >= straight_angle:
            speed = 1.5 # at corners
        else: speed = 4.0 # Straight speed
        
        
        print('Steering angle in degrees: {}'.format((steering_angle/(np.pi/2))*90))
        return speed, steering_angle
    