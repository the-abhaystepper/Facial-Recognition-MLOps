import os

def test_data_path():
    assert os.path.exists('dataset'), "Dataset directory must exist"

def test_model_script():
    assert os.path.exists('train_face_model.py'), "Training script must exist"

def test_csv_logs():
    assert os.path.exists('detection_log.csv'), "Detection log must exist"
