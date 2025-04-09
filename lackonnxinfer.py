import os
import cv2
import numpy as np
import onnxruntime
import time

def process_folder(folder_path, session, input_name):
    if not os.path.isdir(folder_path):
        print("유효한 폴더 경로가 아닙니다.")
        return False

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(".png")]
    if not files:
        print("해당 폴더에 PNG 파일이 없습니다.")
        return False

    t0 = time.time()
    
    def sort_key(filename):
        base, _ = os.path.splitext(filename)
        return int(base) if base.isdigit() else base
    
    files = sorted(files, key=sort_key)
    WIDTH, HEIGHT = 224, 224
    imgs_tensor = []
    imgs_vis = []
    
    t_preprocess_start = time.time()
    for f in files:
        path = os.path.join(folder_path, f)
        img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue
        img_resized = cv2.resize(img_gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        img_float = img_resized.astype(np.float32) / 255.0
        img_tensor = img_float[np.newaxis, :, :]
        imgs_tensor.append(img_tensor)
        img_color = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        imgs_vis.append(img_color)

    if len(imgs_tensor) == 0:
        print("유효한 이미지가 없습니다.")
        return False

    FIXED_SEQ_LEN = 132
    num_frames = len(imgs_tensor)
    if num_frames < FIXED_SEQ_LEN:
        while len(imgs_tensor) < FIXED_SEQ_LEN:
            imgs_tensor.append(imgs_tensor[-1])
            imgs_vis.append(imgs_vis[-1])
    elif num_frames > FIXED_SEQ_LEN:
        imgs_tensor = imgs_tensor[:FIXED_SEQ_LEN]
        imgs_vis = imgs_vis[:FIXED_SEQ_LEN]

    seq = np.stack(imgs_tensor, axis=0)[np.newaxis, ...]
    t_preprocess_end = time.time()

    t_infer_start = time.time()
    outputs = session.run(None, {input_name: seq})
    preds = outputs[0].squeeze(0)
    t_infer_end = time.time()

    OUTPUT_VIDEO_PATH = "./output_inference.mp4"
    FPS = 10
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FPS, (WIDTH, HEIGHT))

    t_vis_start = time.time()
    for i, img in enumerate(imgs_vis):
        x, y = preds[i]
        cv2.circle(img, (int(round(x)), int(round(y))), 5, (0, 0, 255), -1)
        out.write(img)
    out.release()
    t_vis_end = time.time()

    print("Preprocessing time: {:.3f} sec".format(t_preprocess_end - t_preprocess_start))
    print("Inference time: {:.3f} sec".format(t_infer_end - t_infer_start))
    print("Visualization time: {:.3f} sec".format(t_vis_end - t_vis_start))
    print("Total time: {:.3f} sec".format(time.time() - t0))
    return True

def main():
    t0 = time.time()
    OPT_ONNX_MODEL_PATH = "./optimized_model.onnx"
    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = 4
    session_options.inter_op_num_threads = 4
    
    session = onnxruntime.InferenceSession(
        OPT_ONNX_MODEL_PATH,
        sess_options=session_options,
        providers=["CPUExecutionProvider"]
    )
    
    input_name = session.get_inputs()[0].name
    t_model_load = time.time()
    print("Model load time: {:.3f} sec".format(t_model_load - t0))
    
    while True:
        print("\n'q'를 입력하면 프로그램이 종료됩니다.")
        folder_path = input("PNG 파일들이 들어있는 폴더 경로를 입력하세요: ").strip()
        
        if folder_path.lower() == 'q':
            print("프로그램을 종료합니다.")
            break
            
        process_folder(folder_path, session, input_name)

if __name__ == "__main__":
    main()