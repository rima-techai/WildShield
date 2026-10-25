import subprocess,wave
import numpy as np
import cv2
from imageio_ffmpeg import get_ffmpeg_exe
from wildshield.io.audio_video import mux_alert_audio


def test_mux_places_sound_on_video_timeline_and_preserves_original(tmp_path):
    video=tmp_path/'silent.mp4'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),10,(32,32))
    for _ in range(20):writer.write(np.zeros((32,32,3),np.uint8))
    writer.release();original=video.read_bytes()
    sound=tmp_path/'tone.wav'
    with wave.open(str(sound),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(44100)
        t=np.arange(22050)/44100
        f.writeframes((np.sin(2*np.pi*440*t)*12000).astype('<i2').tobytes())
    output=tmp_path/'with_audio.mp4'
    mux_alert_audio(video,[(1.0,sound)],2.0,output)
    decoded=subprocess.run([get_ffmpeg_exe(),'-v','error','-i',str(output),'-map','0:a:0','-f','f32le','-ac','1','-ar','8000','-'],capture_output=True,check=True)
    audio=np.frombuffer(decoded.stdout,dtype='<f4')
    assert np.max(np.abs(audio[:6000]))<.001
    assert np.sqrt(np.mean(audio[8500:11000]**2))>.05
    assert video.read_bytes()==original
    cap=cv2.VideoCapture(str(output));assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==20;cap.release()


def test_project_speech_assets_have_non_silent_samples():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for name in ('person_warning', 'risk_warning', 'risk_danger'):
        result = subprocess.run([get_ffmpeg_exe(), '-v', 'error', '-i', str(root/'assets/sounds'/f'{name}.aiff'), '-f', 'f32le', '-ac', '1', '-ar', '8000', '-'],capture_output=True,check=True)
        samples = np.frombuffer(result.stdout,dtype='<f4')
        assert len(samples)>1000, name
        assert np.max(np.abs(samples))>.01, name
