import argparse
import os
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from gevent import monkey
monkey.patch_all()
from flask import Flask, request
from gevent import pywsgi
from PIL import Image
import uuid

from utils import visualization_utils as vis_util
from utils import label_map_util

FLAGS = tf.app.flags.FLAGS
if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')
tf.app.flags.DEFINE_integer('port', '5002',
    'server with port,if no port, use deault port 80')

tf.app.flags.DEFINE_boolean('debug', False, '')
tf.app.flags.DEFINE_string('static_folder', './static', '')
NUM_CLASSES = 5
ALLOWED_EXTENSIONS = set(['jpg','JPG', 'jpeg', 'JPEG', 'png'])
def allowed_files(filename):
    return '.' in filename and \
    filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
tf.app.flags.DEFINE_string('checkpoint_dir', './try', '')
tf.app.flags.DEFINE_string('dataset_dir', './try', '')
tf.app.flags.DEFINE_string('output_dir', './try', '')
def parse_args(check=True):

    parser = argparse.ArgumentParser()
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed

def rename_filename(old_file_name):
  basename = os.path.basename(old_file_name)
  name, ext = os.path.splitext(basename)
  new_name = str(uuid.uuid1()) + ext
  return new_name

PATH_TO_CKPT = os.path.join(FLAGS.checkpoint_dir, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(FLAGS.dataset_dir, 'bdd_label_map.pbtxt')
app = Flask(__name__)


app._static_folder = FLAGS.static_folder
UPLOAD_FOLDER = FLAGS.static_folder + '/upload'
OUTPUT_FOLDER = FLAGS.static_folder + '/output'
@app.route("/", methods=['GET', 'POST'])

def index():
    output_dir='./try'
    result = """
        <!doctype html>
        <title>车辆识别</title>
        <h1>车辆识别系统模拟</h1>
        <form action="" method=post enctype=multipart/form-data>
        <p><input type=file name=file value='选择图片'>
        
          <br/><br/>
             <input type=submit value='上传预测'>
        </form>
        <p>%s</p>
        """ % "<br>"
    if request.method == 'POST':
        file = request.files['file']
        old_file_name = file.filename
        if file and allowed_files(old_file_name):
            filename = rename_filename(old_file_name)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            out_html = inference(file_path)
            return result + out_html
    return result


def inference(image_path):
    FLAGS, unparsed = parse_args()


    # PATH_TO_LABELS = os.path.join(FLAGS.dataset_dir, 'mscoco_label_map.pbtxt')
    # PATH_TO_LABELS = os.path.join(FLAGS.dataset_dir, 'kitti_label_map.pbtxt')


    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    #test_img_path = os.path.join(FLAGS.dataset_dir, 'test6.jpg')

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            image = Image.open(image_path)
            image_np = load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            print(detection_classes)
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=2)
            filename = rename_filename('output.png')
            plt.imsave(os.path.join(OUTPUT_FOLDER, filename), image_np)

            new_url = OUTPUT_FOLDER + '/%s' % os.path.basename(os.path.join(OUTPUT_FOLDER, filename))
            image_tag = '<img src="%s"></img><p>'
            new_tag = image_tag % new_url
    return new_tag

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=FLAGS.port, debug=FLAGS.debug, threaded=True)

