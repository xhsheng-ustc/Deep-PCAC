import os
import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import MaxPool1D, Layer, BatchNormalization

from pnet2_layers.cpp_modules import (
	farthest_point_sample,
	gather_point,
	query_ball_point,
	group_point,
	knn_point,
	three_nn,
	three_interpolate
)

def sample_and_group(npoint, nsample, xyz, points):

	new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
	_,idx = knn_point(nsample, xyz, new_xyz)
	grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
	grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
	if points is not None:
		grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
		new_points = grouped_points
	else:
		new_points = grouped_xyz
	return new_xyz, new_points
def knn_interpolate(xyz1,xyz2,feature,nsample):
	dist, idx = knn_point(nsample, xyz1, xyz2)
	print(dist)
	print(idx)
	dist = tf.maximum(dist, 1e-10)
	norm = tf.reduce_sum((1.0/dist),axis=2, keepdims=True)
	norm = tf.tile(norm,[1,1,3])
	weight = (1.0/dist) / norm
	

def FPS(npoint, xyz):
	xyz = tf.squeeze(xyz,[2])
	new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
	new_xyz = tf.expand_dims(new_xyz,[2])
	return new_xyz
