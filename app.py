from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import pafy
import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

app = Flask(__name__)

IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64

CLASSES_LIST = ["Archery", "BaseballPitch", "Bowling", "BoxingPunchingBag" 
                ,"CliffDiving","Drumming","GolfSwing",
                "JumpingJack","PlayingGuitar","Punch","WritingOnBoard"]

SEQUENCE_LENGTH = 20

model = load_model('model_file.h5')


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get YouTube video link from the form
        youtube_video_url = request.form['video_url']
        
        # Download the video from YouTube
        video = pafy.new(youtube_video_url)
        title = video.title
        video_best = video.getbest()
        input_file_path = f'{secure_filename(title)}.mp4'
        video_best.download(filepath=f'static/{input_file_path}', quiet=True)
        
        # Predict on the video
        predict_on_video(input_file_path)
        
        # Return the output video file
        output_file_path = f'{os.path.splitext(input_file_path)[0]}-Output-SeqLen{SEQUENCE_LENGTH}.mp4'
        return render_template('result.html', input_video=input_file_path, output_video=output_file_path, classes_list=CLASSES_LIST)
    else:
        return render_template('index.html')


# @app.route('/static', methods=['GET'])
# def getVideo():


def predict_on_video(video_file_path):

    video_reader = cv2.VideoCapture(f'static/{video_file_path}')
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_reader.get(cv2.CAP_PROP_FPS)

    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    predicted_class_name = ''
    output_file_path = f'{os.path.splitext(video_file_path)[0]}-Output-SeqLen{SEQUENCE_LENGTH}.mp4'
    video_writer = cv2.VideoWriter(f'static/_1{output_file_path}', cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), fps, (original_video_width, original_video_height))

    while video_reader.isOpened():

        ok, frame = video_reader.read() 
        
        if not ok:
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_queue.append(normalized_frame)

        if len(frames_queue) == SEQUENCE_LENGTH:

            predicted_labels_probabilities = model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)
            predicted_class_name = CLASSES_LIST[predicted_label]

        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        video_writer.write(frame)
        
    video_reader.release()
    video_writer.release()

    os.system(f"ffmpeg -i static/_1{output_file_path} -vcodec libx264 -y static/{output_file_path}")



if __name__ == '__main__':
    app.run(debug=True)