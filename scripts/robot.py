#Author: YuLong Pei, Zizhou Zhai
#!/usr/bin/env python

import rospy
import random
import math
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Pose, PoseArray
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from read_config import read_config
from helper_functions import get_pose, move_function
from map_utils import *
from sklearn.neighbors import KDTree

class Robot():
	def __init__(self):
		rospy.init_node("robot")

		# Read data from config file
		self.config = read_config()
		self.move_list = self.config["move_list"]
		self.num_particles = self.config["num_particles"]

		self.completed_moves = 0


		# Set up the publishers for the robot
		self.particlecloud_publisher = rospy.Publisher(
			"/particlecloud",
			PoseArray,
			queue_size = 10,
			latch=True
		)

		self.likelihood_field_publisher = rospy.Publisher(
			"/likelihood_field",
			OccupancyGrid,
			queue_size = 10,
			latch=True
		)

		self.result_update_publisher = rospy.Publisher(
			"/result_update",
			Bool,
			queue_size = 10
		)

		self.sim_complete_publisher = rospy.Publisher(
			"/sim_complete",
			Bool,
			queue_size = 10
		)

		# Set up the subscribers for this robot
		self.map_subscriber = rospy.Subscriber(
			"/map",
			OccupancyGrid,
			self.handle_map_message
		)

		self.base_scan_subscriber = rospy.Subscriber(
			"/base_scan",
			LaserScan,
			self.handle_base_scan_message
		)


		rospy.spin()

	def handle_map_message(self, message):


		# Instantiate map
		self.map = Map(message)
		self.orig_map = Map(message)

		# Read data from the map topic
		self.map_occupancygrid = message
		self.map_width = message.info.width
		self.map_height = message.info.height

		# Create the pose_Array
		self.pose_array = PoseArray()
		self.pose_array.header.stamp = rospy.Time.now()
		self.pose_array.header.frame_id = 'map'
		self.pose_array.poses = []

		# Create the particle array
		self.particle_array = []

		# Loop to generate num_particles randomly
		particleCount = 0

		while(particleCount < self.num_particles):
			particle_x = random.random()*self.map_width
			particle_y = random.random()*self.map_height
			particle_theta = random.random()*2*math.pi
			particle_weight = 1./self.num_particles

			(x1,y1) = self.map.cell_position(particle_y, particle_x)

			if(self.map.get_cell(x1,y1) != 1 and not (math.isnan(self.map.get_cell(x1,y1))) ):
				particle_pose = get_pose(particle_x, particle_y, particle_theta);
				new_particle = Particle(particle_x, particle_y, particle_theta, particle_weight, particle_pose)
				self.particle_array.append(new_particle)
				self.pose_array.poses.append(particle_pose)
				particleCount = particleCount + 1


		self.laser_sigma_hit = self.config["laser_sigma_hit"]
		self.zhit = self.config["laser_z_hit"]
		self.zrand = self.config["laser_z_rand"]

		list_of_points = [] #List of obstacles
		list_of_query_points = [] #list of all points on the map

		for i in range(self.map_width):
			for j in range(self.map_height):
				(col,row) = self.map.cell_position(j,i)
				if (self.map.get_cell(col, row) == 1) and not self.is_inner_obstacle(i,j):
					list_of_points.append((col,row))
				list_of_query_points.append((col,row))

		kdt = KDTree(list_of_points)
		(dists,indices) = kdt.query(list_of_query_points, k=1)

		# populate new map with likelihood values
		index = 0
		for i in range(self.map_width):
			for j in range(self.map_height):
				(col,row) = self.map.cell_position(j,i)
				dist = self.likelihood_prob(dists[index],self.laser_sigma_hit)

				self.map.set_cell(col,row,dist)
				index += 1;
		self.particlecloud_publisher.publish(self.pose_array)

		self.likelihood_field_publisher.publish(self.map.to_message())

		rospy.sleep(1)

		# Published particles and likelihood function. Start movement.
		self.current_move_step_count = 0
		self.current_move_step_required = 0

		while len(self.move_list) > 0:
			b = Bool()
			b.data = True
			self.result_update_publisher.publish(b)

			self.move_robot()
			#self.particlecloud_publisher.publish(self.pose_array)
			

		b = Bool()
		b.data = True

		self.sim_complete_publisher.publish(b)
		rospy.sleep(1)
		rospy.signal_shutdown("Simulation complete")


	def reweight_particles(self):
		self.normalize_sum = 0

		for particle in self.particle_array:
			self.current_angle = self.recent_laser_results.angle_min
			self.ptotal = 0
			if(particle.weight == 0):
				continue;
			will_del = 0

			for laser_dist in self.recent_laser_results.ranges:
				if(laser_dist > self.recent_laser_results.range_min and laser_dist < self.recent_laser_results.range_max):
					# calculate new location
					self.endpoint = (particle.x+laser_dist*math.cos(self.current_angle+particle.theta),particle.y+laser_dist*math.sin(self.current_angle+particle.theta))
					# get likelihood of new location

					(x1,y1) = self.map.cell_position(self.endpoint[1],self.endpoint[0])

					if not (math.isnan(self.map.get_cell(x1,y1))):
						pz = self.zhit*self.map.get_cell(x1,y1)+self.zrand
						# sum pz into ptotal
						self.ptotal += math.pow(pz,3)
					elif math.isnan(self.map.get_cell(x1,y1)) or self.map.get_cell(x1,y1) == 0:
						will_del = will_del + 1


				self.current_angle += self.recent_laser_results.angle_increment

			if will_del > 20:
				particle.weight = 0
			else:
				particle.weight = particle.weight* (1/(1+math.exp(-self.ptotal)))
				#particle.weight = particle.weight*self.ptotal
				self.normalize_sum += particle.weight

		totalsum = 0.0
		for particle in self.particle_array:
			particle.weight = particle.weight/self.normalize_sum
			totalsum+= particle.weight
		print "total: " + str(totalsum)

	def resample_particles(self):
		self.new_particle_array = []
		self.pose_array.poses = []

		for turns in range(self.num_particles):
			resample_value = random.random()
			for particle in self.particle_array:
				resample_value -= particle.weight
				if(resample_value <= 0):
					particle_x = particle.x + random.gauss(0, self.config["resample_sigma_x"])
					particle_y = particle.y + random.gauss(0, self.config["resample_sigma_y"])
					particle_theta = particle.theta + random.gauss(0, self.config["resample_sigma_angle"])
					particle_pose = get_pose(particle_x, particle_y, particle_theta)
					self.new_particle = Particle(particle_x, particle_y, particle_theta, particle.weight, particle_pose)
					
					self.new_particle_array.append(self.new_particle)
					self.pose_array.poses.append(particle_pose)

					break;

		self.particle_array = self.new_particle_array




	def handle_base_scan_message(self, message):
		self.recent_laser_results = message;
		pass

	def move_robot(self):


		self.current_move = self.move_list.pop(0)
		self.current_move_step_count = 0
		self.current_move_step_required = self.current_move[2]


		# Turn the robot one time
		move_function(self.current_move[0], 0)


		# Change degrees to radian
		self.degree = self.current_move[0]
		self.radian = self.degree * math.pi / 180


		self.rotate_each_particle(self.radian)



		while (self.current_move_step_count < self.current_move_step_required):
			# Move the robot forward
			self.current_move_step_count += 1
			move_function(0, self.current_move[1])

			if(self.completed_moves == 0):
				# add noise to each particle
				self.add_noise_particle()


			# Perform move update for all particles
			self.move_update_particle(self.current_move[1])


			self.remove_out_of_bound();


			self.reweight_particles();

			self.ignore_low_weight();


			self.resample_particles();

			self.reweight_particles();

			self.ignore_low_weight();


			self.resample_particles();


			self.particlecloud_publisher.publish(self.pose_array)



		self.completed_moves += 1
		self.particlecloud_publisher.publish(self.pose_array)


	def ignore_low_weight(self):
		weight_array = []
		for particle in self.particle_array:
			weight_array.append(particle.weight)
		weight_array.sort()
		index = (self.num_particles/10) -1
		weight_threshold = weight_array[index]
		for particle in self.particle_array:
			if particle.weight <= weight_threshold:
				particle.weight = 0;


	def remove_out_of_bound(self):
		for particle in self.particle_array:
			(x1,y1) = self.map.cell_position(particle.y,particle.x)
			if math.isnan(self.orig_map.get_cell(x1, y1)):
				particle.weight = 0.0
			if self.orig_map.get_cell(x1,y1) == 1:
				particle.weight = 0.0


	def rotate_each_particle(self, rotate_amount):

		for particle in self.particle_array:
			particle.theta += rotate_amount


	def move_update_particle(self, distance):

		for particle in self.particle_array:
			particle.x += distance*math.cos(particle.theta)
			particle.y += distance*math.sin(particle.theta)


	def add_noise_particle(self):

		self.pose_array.poses = []

		for particle in self.particle_array:
			particle.x += random.gauss(0, self.config["first_move_sigma_x"])
			particle.y += random.gauss(0, self.config["first_move_sigma_y"])
			particle.theta += random.gauss(0, self.config["first_move_sigma_angle"])
			particle.pose = get_pose(particle.x, particle.y, particle.theta)
			self.pose_array.poses.append(particle.pose)


	def likelihood_prob(self, dist, sigma):
		#return (1./(math.sqrt(2*math.pi)*sigma))*math.exp(-math.pow(dist,2)/(2*math.pow(sigma,2)))

		return math.exp(-math.pow(dist,2)/(2*math.pow(sigma,2)))

	def is_inner_obstacle(self, i,j):

		threshold = 1

		(col,row) = self.orig_map.cell_position(j,i)

		if (math.isnan(self.orig_map.get_cell(col-threshold,row))):
			return True
		if (math.isnan(self.orig_map.get_cell(col+threshold,row))):
			return True
		if (math.isnan(self.orig_map.get_cell(col,row+threshold))):
			return True
		if (math.isnan(self.orig_map.get_cell(col,row-threshold))):
			return True
		if (math.isnan(self.orig_map.get_cell(col,row))):
			return True

		if (self.orig_map.get_cell(col-threshold,row) != 1):
			return False

		if (self.orig_map.get_cell(col+threshold,row) != 1) :
			return False

		if (self.orig_map.get_cell(col,row+threshold) != 1) :
			return False

		if (self.orig_map.get_cell(col,row-threshold) != 1) :
			return False

		if (self.orig_map.get_cell(col,row) != 1) :
			return False

		else:
			return True

class Particle:
	def __init__(self,x,y,theta,weight,pose):
		self.x = x
		self.y = y
		self.theta = theta
		self.weight = weight
		self.pose = pose


if __name__ == '__main__':
	r = Robot()
