# Imported Python Transfer Function
import numpy as np
from hbp_nrp_cle.robotsim.RobotInterface import Topic
import geometry_msgs.msg
import cv2
#@nrp.MapSpikeSink('line_recorder', nrp.brain.actors, nrp.spike_recorder)
@nrp.MapRobotSubscriber("lastTerm_sub", Topic("robot/lastTermTime", std_msgs.msg.Float64))
@nrp.MapRobotPublisher("lastTerm_writer", Topic("robot/lastTermTime", std_msgs.msg.Float64))
@nrp.MapRobotSubscriber("invert", Topic("robot/invertStart", std_msgs.msg.Bool))
@nrp.MapRobotPublisher("invert_writer", Topic("robot/invertStart", std_msgs.msg.Bool))
@nrp.MapRobotSubscriber("offsetHead", Topic("/alphaS/offsetHead", std_msgs.msg.Float64))
@nrp.MapRobotSubscriber("offset", Topic("/alphaS/offset", std_msgs.msg.Float64))
@nrp.MapRobotSubscriber("type", Topic("/alphaS/gait_type", std_msgs.msg.String))
@nrp.MapRobotPublisher("joint_16", Topic("/alphaS/joint_16/cmd_pos", std_msgs.msg.Float64))
@nrp.MapRobotPublisher("joint_15", Topic("/alphaS/joint_15/cmd_pos", std_msgs.msg.Float64))
@nrp.MapRobotPublisher("joint_14", Topic("/alphaS/joint_14/cmd_pos", std_msgs.msg.Float64))
@nrp.MapRobotPublisher("joint_13", Topic("/alphaS/joint_13/cmd_pos", std_msgs.msg.Float64))
@nrp.MapRobotPublisher("joint_12", Topic("/alphaS/joint_12/cmd_pos", std_msgs.msg.Float64))
@nrp.MapRobotPublisher("joint_11", Topic("/alphaS/joint_11/cmd_pos", std_msgs.msg.Float64))
@nrp.MapRobotPublisher("joint_10", Topic("/alphaS/joint_10/cmd_pos", std_msgs.msg.Float64))
@nrp.MapRobotPublisher("joint_9", Topic("/alphaS/joint_9/cmd_pos", std_msgs.msg.Float64))
@nrp.MapRobotPublisher("joint_8", Topic("/alphaS/joint_8/cmd_pos", std_msgs.msg.Float64))
@nrp.MapRobotPublisher("joint_7", Topic("/alphaS/joint_7/cmd_pos", std_msgs.msg.Float64))
@nrp.MapRobotPublisher("joint_6", Topic("/alphaS/joint_6/cmd_pos", std_msgs.msg.Float64))
@nrp.MapRobotPublisher("joint_5", Topic("/alphaS/joint_5/cmd_pos", std_msgs.msg.Float64))
@nrp.MapRobotPublisher("joint_4", Topic("/alphaS/joint_4/cmd_pos", std_msgs.msg.Float64))
@nrp.MapRobotPublisher("joint_3", Topic("/alphaS/joint_3/cmd_pos", std_msgs.msg.Float64))
@nrp.MapRobotPublisher("joint_2", Topic("/alphaS/joint_2/cmd_pos", std_msgs.msg.Float64))
@nrp.MapRobotPublisher("joint_1", Topic("/alphaS/joint_1/cmd_pos", std_msgs.msg.Float64))
@nrp.Neuron2Robot()
def snake_control(t, invert, invert_writer, lastTerm_writer, lastTerm_sub, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7, joint_8, joint_9, joint_10, joint_11, joint_12, joint_13, joint_14, joint_15, joint_16, type, offset, offsetHead):
	if t < 1.:
		return

	gait_type = std_msgs.msg.String("slithering") # rolling, slithering, rollingTurn, scanning, none
	joints = [None, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7, joint_8, joint_9, joint_10, joint_11, joint_12, joint_13, joint_14, joint_15, joint_16]
	N	= len(joints)      
	C_o=std_msgs.msg.Float64(0.0 * np.pi / 180.0).data
	C_oH = C_o
	lastTimeReset = 0.
	inverting = False
		
	if invert.value is None:
		invert_writer.send_message(std_msgs.msg.Bool(True))
	else:
		inverting = invert.value.data

	if lastTerm_sub.value is None:
		lastTerm_writer.send_message(std_msgs.msg.Float64(0.0))
	else:	
		lastTimeReset = lastTerm_sub.value.data
	if t < lastTimeReset + 0.02:
		invert_writer.send_message(std_msgs.msg.Bool(not inverting))


	# ------------Manipulate Time
	if t < (lastTimeReset + 1.0): 
		for i in range(1, N):
			joints[i].send_message(std_msgs.msg.Float64(0))
		return
	else:
		add = 6.24
		if not inverting:
			add = 0
		if t < (lastTimeReset + 1.5):
			t = 0 + add
		else:
			t = max(t - lastTimeReset - 1.5, 0)
			t = t + add
			t = t % 12.5

	
	# jointAngels = []
	# for i in range(N):
	# 	jointAngels.append(0.)

	if type.value is not None:
		gait_type = type.value
	if offset.value is not None:
		C_o = offset.value.data
	if offsetHead.value is not None:
		C_oH = offsetHead.value.data
	
		
	if gait_type == std_msgs.msg.String("rolling"):	#rolling gait
		amp 	= 15.0*np.pi/180.0
		omega 	= np.pi
		phi_diff= 0.5*np.pi  
        	d 	= 1	# direction (1 or -1)
		for i in range(1, N):
			joints[i].send_message(std_msgs.msg.Float64(amp*np.sin(omega*t*d + phi_diff*(i-1))))
	elif gait_type == std_msgs.msg.String("slithering"):	#slithering gait
		# spatial frequency
		spat_freq_o = 6.0/float(N)*np.pi
		spat_freq_e = 12.0/float(N)*np.pi
        	#spat_freq_o = 5.0/float(N)*np.pi
		#spat_freq_e = 11.0/float(N)*np.pi

		#temporal frequency
		temp_freq_o = 0.25/np.pi #1.5/np.pi
		temp_freq_e = 0.5/np.pi #3.0/np.pi
        
		# temporal phase offset between horizontal and vertical waves
		TPO = -90.0*np.pi/180.0

		# amplitude
		A_o = 70.0*np.pi/180.0
		A_e = 10.0*np.pi/180.0

		# offset
		#C_o = 0.0*np.pi/180.0
		C_e = 0.0*np.pi/180.0

		#linear coefficient
		z = 0.5
		y = 1-z
		#z_o = 0.3
		#y_o = 1-z_o
		#z_e = 0.75
		#y_e = 1-z_e

		# direction (1 or -1)
		d   = 1	

		# t = t % 1000

		for i in range(1, N):
			if i%2 == 1:
				if i==1: #motion compensation
					#joints[i].send_message(std_msgs.msg.Float64((0.9128*np.sin(0.5027*t+45.0/180.0*np.pi)+0.0827) ))
					#joints[i].send_message(std_msgs.msg.Float64((0.7728*np.sin(0.5027*t+20.0/180.0*np.pi)+0.0677) ))
					
					
					###C_head in -0.5*PI to 0.5PI
					joints[i].send_message(std_msgs.msg.Float64(((i-1)/float(N)*y+z)*A_o*np.sin(2.0*np.pi*temp_freq_o*d * t +
                                                                                 i*spat_freq_o + TPO) + C_o + 0.7728*np.sin(0.5020*t+65.0/180.0*np.pi) + C_oH*0.5*np.pi))  # + (C_head) ))
					# joints[i].send_message(std_msgs.msg.Float64(C_oH))  # + (C_head) ))
				# elif i== 3:
				# 	joints[i].send_message(std_msgs.msg.Float64( ( (i-1)/float(N)*y+z)*A_o*np.sin( 2.0*np.pi*temp_freq_o*d*t + i*spat_freq_o + TPO) + C_o))
				else:
					joints[i].send_message(std_msgs.msg.Float64( ( (i-1)/float(N)*y+z)*A_o*np.sin( 2.0*np.pi*temp_freq_o*d*t + i*spat_freq_o + TPO) + C_o))
			else:
				# if i == 2:
				# 	joints[i].send_message(std_msgs.msg.Float64(((i-1)/float(N)*y+z)*A_e*np.sin(2.0*np.pi*temp_freq_e*d*t + i*spat_freq_e) + C_e))
				# else:
					joints[i].send_message(std_msgs.msg.Float64(((i-1)/float(N)*y+z)*A_e*np.sin(2.0*np.pi*temp_freq_e*d*t + i*spat_freq_e) + C_e))
	elif gait_type == std_msgs.msg.String("rollingTurn"): #rolling_turn gait
		amp	= 15.0*np.pi/180.0
		omega	= np.pi
		phi_diff= 0.5*np.pi
		d_front	= -1
		d_middle= 0
		d_back	= 1
		for i in range(1, N):
			if i < N/2+1:
				joints[i].send_message(std_msgs.msg.Float64(d_front*amp*np.sin(omega*t*d_front + phi_diff*(i-1))))
			elif i > N/2+1:
				joints[i].send_message(std_msgs.msg.Float64(d_back*amp*np.sin(omega*t*d_back + phi_diff*(i-1))))
			else:
				joints[i].send_message(std_msgs.msg.Float64(d_middle*amp*np.sin(omega*t*d_middle + phi_diff*(i-1))))
	elif gait_type == std_msgs.msg.String("scanning"): #scanning gait
		head_amp	= 50.0*np.pi/180.0
		head_omega	= 0.25*np.pi
		for i in range(1, N):
			if i == 1:
				joints[i].send_message(std_msgs.msg.Float64(head_amp*np.sin(head_omega*t)))
			if i == 2:
				joints[i].send_message(std_msgs.msg.Float64(80.0*np.pi/180.0))
			if i == 3:
				joints[i].send_message(std_msgs.msg.Float64(0.0))
			if i == 4:
				joints[i].send_message(std_msgs.msg.Float64(80.0*np.pi/180.0))
			if i == 5:
				joints[i].send_message(std_msgs.msg.Float64(30.0*np.pi/180.0))
			if i > 5 and i%2 == 1:
				joints[i].send_message(std_msgs.msg.Float64(80.0*np.pi/180.0))
			if i > 5 and i%2 == 0:
				joints[i].send_message(std_msgs.msg.Float64(0.0))
	else:
		pass
	
