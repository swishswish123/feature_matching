#### RUN THIS TO CREATE IMAGE FRAMES FROM VIDEO
from urllib.error import ContentTooShortError
from PIL import Image
import os
import glob
import cv2
from pathlib import Path



def convert_vid_frames(vid_path, save_folder, start_frame=0, end_frame=False, pat_id='051', pat_seg='00'):
    cap = cv2.VideoCapture(vid_path)
    totalframecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'{totalframecount} total frames. Start {start_frame}, end {end_frame}')

    print('before')
    if not os.path.isdir(save_folder):
        print(f'creating {save_folder}')
        os.mkdir(f'{save_folder}')

    frame_num = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if frame_num < start_frame:
            continue
        elif end_frame :
            if frame_num == end_frame:
                break
        # turn to gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frames', gray)
        if cv2.waitKey(1) == ord('q'):
            break
        frame_num += 1

        cv2.imwrite('{}/{:08d}.png'.format(save_folder, frame_num), frame)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    return


def main():
    # project_path='/Users/aure/Desktop/i4health/project/endoSim/endosim/DenseDescriptorLearning-Pytorch-matt'
    # data_path = f'{project_path}/data'
    # os.path.join(data_path, '2/051_01_trimmed.mp4')
    # create_patient_folder(data_path)
    # convert_vid_frames(data_path)
    project_path = Path(__file__).parent.resolve()

    vid_paths = glob.glob(f'{project_path}/dataset/endo_videos/*/*.mpg')

    if not os.path.isdir(f'{project_path}/dataset/endo_data'):
        print(f'creating {project_path}/dataset/endo_data')
        os.mkdir(f'{project_path}/dataset/endo_data')

    for seq_num, vid_path in enumerate(vid_paths):
        # data_path = f'{project_path}/data'
        save_folder = f'{project_path}/dataset/endo_data/seq_{seq_num}'
        print('here')
        convert_vid_frames(vid_path, save_folder, end_frame=False)

    '''
    start_frame = 0
    end_frame = 100
    frame_rate = 1
    img_path = f'/endosim/DenseDescriptorLearning-Pytorch/data/2/_start_{start_frame}_end_{end_frame}_stride_{frame_rate}_segment_00/images_noncrop'
    '''


if __name__ == '__main__':
    main()

    '''
    # resize and save images
    for image_path in os.listdir(img_path):
        # check if the image ends with png
        if (image_path.endswith(".jpg")):
            # image = Image.open('/Users/aure/Desktop/i4health/project/endoSim/endosim/DenseDescriptorLearning-Pytorch/data/1/_start_002603_end_002984_stride_1000_segment_00/images/00002603.jpg')
            image = Image.open(f'{img_path}/{image_path}')
            new_image = image.resize((320, 256))
            image.save(f'{cropped_img_path}/{image_path}')
    '''