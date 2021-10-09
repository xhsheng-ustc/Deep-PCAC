import os
import argparse
import numpy as np
import tensorflow as tf
import importlib 
import subprocess
tf.enable_eager_execution()

from entropy_model import EntropyBottleneck
from conditional_entropy_model import SymmetricConditional
import open3d as o3d
###################################### Preprocess & Postprocess ######################################
def preprocess(input_file, points_num=2048):
  """Partition.
  Input: .ply file and arguments for pre-process.  
  Output: partitioned cubes, cube positions, and number of points in each cube. 
  """

  print('===== Partition =====')
  # scaling (optional)
  pcd = o3d.io.read_point_cloud(input_file)
  coordinate = np.asarray(pcd.points)
  color = np.asarray(pcd.colors)
  point_cloud = np.concatenate((coordinate,color),axis=1)
  number_of_points_of_ply = point_cloud.shape[0]
  number_of_feature = point_cloud.shape[1]
  set_num  = int(np.ceil(number_of_points_of_ply/points_num))
  point_set = np.zeros((1,points_num,number_of_feature))
  point_cloud = np.expand_dims(point_cloud,0)

  for i in range(set_num):
    if i <set_num-1:
      #print(i)
      point_set = np.concatenate((point_set,point_cloud[:,i*2048:(i+1)*2048,:]),0)
    else:
      temp  = np.zeros((1,points_num,number_of_feature))
      num_less_than_2048 = number_of_points_of_ply-points_num*i
      #number points of last set whose number of points is less than 2048
      temp[:,0:num_less_than_2048,:] = point_cloud[:,i*points_num:,:]
      point_set = np.concatenate((point_set,temp),0)
  point_set = point_set[1:,:,:]
  print(point_set.shape)
  print("Partition")
  return point_set,num_less_than_2048

def postprocess(output_file, point_set, num_less_than_2048,points_num=2048):
  """Reconstrcut point cloud and write to ply file.
  Input:  output_file, point_set
  """
  set_num = point_set.shape[0]
  feature_num = point_set.shape[2]
  number_of_points_of_ply = (set_num-1)*points_num+num_less_than_2048
  point_cloud = np.zeros((number_of_points_of_ply,feature_num))
  for i in range(set_num):
    if i<set_num-1:
      point_cloud[i*2048:(i+1)*2048] = point_set[i]
    else:
      point_cloud[i*2048:] = point_set[i,0:num_less_than_2048,:]
  pcd = o3d.geometry.PointCloud()
  point_ori_position = point_cloud[:,0:3]
  point_ori_color = point_cloud[:,3:6]
  pcd.points=o3d.utility.Vector3dVector(point_ori_position)
  pcd.colors=o3d.utility.Vector3dVector(point_ori_color)
  o3d.io.write_point_cloud(output_file,pcd,write_ascii=False)
  return point_cloud

###################################### Compress & Decompress ######################################

def compress(x_coori,x_color,model, ckpt_dir, latent_points):
  """Compress cubes to bitstream.
  Input: cubes with shape [batch size, length, width, height, channel(1)].
  Input: cubes with shape [batch size, num_points=2048, num_feature=6].
  Output: compressed bitstream.
  """

  print('===== Compress =====')
  # load model.
  model = importlib.import_module(model)
  analysis_transform = model.AnalysisTransform(latent_points)
  hyper_encoder = model.HyperEncoder()
  hyper_decoder = model.HyperDecoder()
  entropy_bottleneck = EntropyBottleneck()
  conditional_entropy_model = SymmetricConditional()

  checkpoint = tf.train.Checkpoint(analysis_transform=analysis_transform, 
                                    hyper_encoder=hyper_encoder, 
                                    hyper_decoder=hyper_decoder, 
                                    estimator=entropy_bottleneck)
  status = checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))

  x = tf.convert_to_tensor(x_color, "float32")
  x_coori = tf.convert_to_tensor(x_coori, "float32")

  def loop_analysis(element):
    x = tf.expand_dims(element[0], 0)
    x_coori = tf.expand_dims(element[1], 0)
    y = analysis_transform(x_coori,x)
    return tf.squeeze(y,axis=0)

  element = [x,x_coori]
  ys = tf.map_fn(loop_analysis, element, dtype=tf.float32, parallel_iterations=1, back_prop=False)
  print("Analysis Transform")

  def loop_hyper_encoder(y):
    y = tf.expand_dims(y, 0)
    z = hyper_encoder(y)
    return tf.squeeze(z,axis=0)

  zs = tf.map_fn(loop_hyper_encoder, ys, dtype=tf.float32, parallel_iterations=1, back_prop=False)
  print("Hyper Encoder")

  z_hats, _ = entropy_bottleneck(zs, False)
  print("Quantize hyperprior")

  def loop_hyper_deocder(z):
    z = tf.expand_dims(z, 0)
    loc, scale = hyper_decoder(z)
    return tf.squeeze(loc, [0]), tf.squeeze(scale, [0])

  locs, scales = tf.map_fn(loop_hyper_deocder, z_hats, dtype=(tf.float32, tf.float32),
                          parallel_iterations=1, back_prop=False)
  lower_bound = 1e-9# TODO
  scales = tf.maximum(scales, lower_bound)
  print("Hyper Decoder")

  z_strings, z_min_v, z_max_v = entropy_bottleneck.compress(zs)
  z_shape = tf.shape(zs)[:]
  print("Entropy Encode (Hyper)")

  y_strings, y_min_v, y_max_v = conditional_entropy_model.compress(ys, locs, scales)
  y_shape = tf.shape(ys)[:]
  print("Entropy Encode")

  return y_strings, y_min_v, y_max_v, y_shape, z_strings, z_min_v, z_max_v, z_shape

def decompress(x_coori,y_strings, y_min_v, y_max_v, y_shape, z_strings, z_min_v, z_max_v, z_shape, model, ckpt_dir,latent_points):
  """Decompress bitstream to cubes.
  Input: compressed bitstream. latent representations (y) and hyper prior (z).
  Output: cubes with shape [batch size, length, width, height, channel(1)]
  """

  print('===== Decompress =====')
  # load model.
  model = importlib.import_module(model)
  synthesis_transform = model.SynthesisTransform(latent_points)
  hyper_encoder = model.HyperEncoder()
  hyper_decoder = model.HyperDecoder()
  entropy_bottleneck = EntropyBottleneck()
  conditional_entropy_model = SymmetricConditional()

  checkpoint = tf.train.Checkpoint(synthesis_transform=synthesis_transform, 
                                    hyper_encoder=hyper_encoder, 
                                    hyper_decoder=hyper_decoder, 
                                    estimator=entropy_bottleneck)
  status = checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))

  zs = entropy_bottleneck.decompress(z_strings, z_min_v, z_max_v, z_shape, z_shape[-1])
  print("Entropy Decoder (Hyper)")

  def loop_hyper_deocder(z):
    z = tf.expand_dims(z, 0)
    loc, scale = hyper_decoder(z)
    return tf.squeeze(loc, [0]), tf.squeeze(scale, [0])

  locs, scales = tf.map_fn(loop_hyper_deocder, zs, dtype=(tf.float32, tf.float32),
                          parallel_iterations=1, back_prop=False)
  lower_bound = 1e-9# TODO
  scales = tf.maximum(scales, lower_bound)
  print("Hyper Decoder")

  ys = conditional_entropy_model.decompress(y_strings, locs, scales, y_min_v, y_max_v, y_shape)
  print("Entropy Decoder")

  def loop_synthesis(element):
    y = tf.expand_dims(element[0], 0)
    x_coori = tf.expand_dims(element[1], 0)
    x_coori= tf.cast(x_coori,tf.float32)
    x = synthesis_transform(x_coori,y)
    return tf.squeeze(x, [0])

  element=[ys,x_coori]
  xs = tf.map_fn(loop_synthesis, element, dtype=tf.float32, parallel_iterations=1, back_prop=False)
  print("Synthesis Transform")

  return xs

###################################### write & read binary files. ######################################

def write_binary_files(filename, y_strings, z_strings, points_numbers_less_than2048, y_min_v, y_max_v, y_shape, z_min_v, z_max_v, z_shape, rootdir='/code'):
  """Write compressed binary files:
    1) Compressed latent features.
    2) Compressed hyperprior.
    3) Number of input points.
  """ 

  if not os.path.exists(rootdir):
    os.makedirs(rootdir)
  print('===== Write binary files =====')
  file_strings = os.path.join(rootdir, filename+'.strings')
  file_strings_hyper = os.path.join(rootdir, filename+'.strings_hyper')
  file_pointnums = os.path.join(rootdir, filename+'.pointnums')
  
  with open(file_strings, 'wb') as f:
    f.write(np.array(y_shape, dtype=np.int16).tobytes())# [batch size, length, width, height, channels]
    f.write(np.array((y_min_v, y_max_v), dtype=np.int8).tobytes())
    f.write(y_strings)

  with open(file_strings_hyper, 'wb') as f:
    f.write(np.array(z_shape, dtype=np.int16).tobytes())# [batch size, length, width, height, channels]
    f.write(np.array((z_min_v, z_max_v), dtype=np.int8).tobytes())
    f.write(z_strings)

  # TODO: Compress numbers of points.
  with open(file_pointnums, 'wb') as f:
    f.write(np.array(points_numbers_less_than2048, dtype=np.uint16).tobytes())
    
  bytes_strings = os.path.getsize(file_strings)
  bytes_strings_hyper = os.path.getsize(file_strings_hyper)
  bytes_pointnums = os.path.getsize(file_pointnums)

  print('Total file size (Bytes): {}'.format(bytes_strings+bytes_strings_hyper+bytes_pointnums))
  print('Strings (Bytes): {}'.format(bytes_strings))
  print('Strings hyper (Bytes): {}'.format(bytes_strings_hyper))
  print('Numbers of points (Bytes): {}'.format(bytes_pointnums))

  return bytes_strings, bytes_strings_hyper, bytes_pointnums

def read_binary_files(filename, rootdir='/code'):
  """Read from compressed binary files:
    1) Compressed latent features.
    2) Compressed hyperprior.
    3) Number of input points.
  """ 

  print('===== Read binary files =====')
  file_strings = os.path.join(rootdir, filename+'.strings')
  file_strings_hyper = os.path.join(rootdir, filename+'.strings_hyper')
  file_pointnums = os.path.join(rootdir, filename+'.pointnums')
  
  with open(file_strings, 'rb') as f:
    y_shape = np.frombuffer(f.read(2*4), dtype=np.int16)
    y_min_v, y_max_v = np.frombuffer(f.read(1*2), dtype=np.int8)
    y_strings = f.read()

  with open(file_strings_hyper, 'rb') as f:
    z_shape = np.frombuffer(f.read(2*4), dtype=np.int16)
    z_min_v, z_max_v = np.frombuffer(f.read(1*2), dtype=np.int8)
    z_strings = f.read()

  with open(file_pointnums, 'rb') as f:
    points_numbers_less_than2048 = np.frombuffer(f.read(2), dtype=np.uint16)
  
  return y_strings, z_strings, points_numbers_less_than2048, y_min_v, y_max_v, y_shape, z_min_v, z_max_v, z_shape



def parse_args():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      "command", choices=["compress", "decompress"],
      help="What to do: 'compress' reads a point cloud (.ply format) "
          "and writes compressed binary files. 'decompress' "
          "reads binary files and reconstructs the point cloud (.ply format). "
          "input and output filenames need to be provided for the latter. ")
  parser.add_argument(
      "--input", default='',dest="input",
      help="Input filename.")
  parser.add_argument(
      "--output", default='',dest="output",
      help="Output filename.")
  parser.add_argument(
    "--ckpt_dir", type=str, default='', dest="ckpt_dir",
    help='checkpoint direction trained with different RD tradeoff')
  parser.add_argument(
      "--model", default="model",
      help="model.")
  parser.add_argument(
      "--gpu", type=int, default=1, dest="gpu",
      help="use gpu (1) or not (0).")
  parser.add_argument(
      "--latent_points", type=int, default=256, dest="latent_points")
  args = parser.parse_args()
  print(args)

  return args

if __name__ == "__main__":

  args = parse_args()
  if args.gpu==1:
    os.environ['CUDA_VISIBLE_DEVICES']="0"
  else:
    os.environ['CUDA_VISIBLE_DEVICES']=""
  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 1.0
  config.gpu_options.allow_growth = True
  config.log_device_placement=True
  sess = tf.Session(config=config)

  if args.command == "compress":
    rootdir, filename = os.path.split(args.input)
    if not args.output:
      args.output = filename.split('.')[0]
      print(args.output)
    point_set,num_less_than_2048 = preprocess(args.input)
    x_coori = point_set[:,:,0:3]
    x_color = point_set[:,:,3:6]
    y_strings, y_min_v, y_max_v, y_shape, z_strings, z_min_v, z_max_v, z_shape = compress(x_coori,x_color, args.model, args.ckpt_dir,args.latent_points)

    bytes_strings, bytes_strings_hyper, bytes_pointnums = write_binary_files(
          args.output, y_strings.numpy(), z_strings.numpy(), num_less_than_2048,
          y_min_v.numpy(), y_max_v.numpy(), y_shape.numpy(), 
          z_min_v.numpy(), z_max_v.numpy(), z_shape.numpy(), rootdir='./compressed')

  elif args.command == "decompress":
    rootdir, filename = os.path.split(args.input)
    if not args.output:
      args.output = filename + "_rec.ply"
    ori_cooridinate_path = args.input + ".ply"
    y_strings_d, z_strings_d, num_less_than_2048_d, \
      y_min_v_d, y_max_v_d, y_shape_d, z_min_v_d, z_max_v_d, z_shape_d = read_binary_files(filename, './compressed')

    point_set_ori,num_less_than_2048 = preprocess(ori_cooridinate_path)
    ori_coori = point_set_ori[:,:,0:3]

    rec_color = decompress(ori_coori,y_strings_d, y_min_v_d, y_max_v_d, y_shape_d, z_strings_d, z_min_v_d, z_max_v_d, z_shape_d, args.model, args.ckpt_dir,args.latent_points)
    ori_coori = point_set_ori[:,:,0:3]
    rec_point_cloud = np.concatenate((ori_coori,rec_color),-1)
    postprocess(args.output, rec_point_cloud, int(num_less_than_2048_d),points_num=2048)
    