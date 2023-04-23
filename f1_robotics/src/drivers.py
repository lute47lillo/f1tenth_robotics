import numpy as np
    
class FollowTheGap:    

    PREPROCESS_CONV_SIZE = 3
    MAX_LIDAR_DIST = 3000000
    
    # Set a bubble range of points for the LiDAR to 0 (Safety bubble)
    def create_bubble(self, closest, ranges):
        
        start = closest - 150
        end = closest + 150
        
        if start < 0: 
            start = 0
            
        if end >= len(ranges):
            end = len(ranges)-1
            
        ranges[start:end] = 0
        return ranges
        
    def process_lidar(self, ranges, actions):
    
        _, speed_ppo = actions
        
        # The angles between each of the LiDAR points
        angle_LiDAR = (2*np.pi) / len(ranges)
        
        # Delete LiDAR points behind the vehicle
        ranges = np.array(ranges[135:-135])
        
        # sets each value to the mean over a given window
        ranges = np.convolve(ranges, np.ones(3), 'same') / 3
        ranges = np.clip(ranges, 0, 3000000)
        
        #Find closest point to LiDAR
        closest_point = ranges.argmin()
        
        # Create safety bubble range
        range_bubble = self.create_bubble(closest_point, ranges)

        # Mask the range bubble (set to 0 -> False)
        mask = np.ma.masked_where(range_bubble==0, range_bubble)
        
        # Get contigous gap sequence of non-bubble data (not masked)
        contiguous_gap = np.ma.notmasked_contiguous(mask)
        
        # Get gap_ranges of contiguous data
        start_gap = contiguous_gap[0].start
        end_gap = contiguous_gap[0].stop
        gap_ranges = range_bubble[start_gap:end_gap]
        
        #Find the best point in the gap 
        averaged_max_gap = np.convolve(gap_ranges, np.ones(80), 'same') / 80
        best_gap = averaged_max_gap.argmax() + start_gap

        # Get Actions speed and steering angle from reactive method follow the gap
        scan_angle = (best_gap - (len(range_bubble)/2)) * angle_LiDAR
        steering_angle = scan_angle / 2
  
        angle_straight = 1.5708
         
        # Corner speed
        if abs(steering_angle) > angle_straight:
            speed = speed_ppo/2
        else: # Straight line speed
            speed = speed_ppo
        
        return speed, steering_angle
