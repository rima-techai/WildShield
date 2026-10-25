"""Embed generated alert events in the recorded frame timeline (not microphone audio)."""
from pathlib import Path
import os
import subprocess
import tempfile


def mux_alert_audio(video: Path, events: list[tuple[float, Path]], duration: float, output: Path | None = None) -> Path:
    from imageio_ffmpeg import get_ffmpeg_exe
    if duration <= 0:
        raise ValueError('Video duration must be positive')
    events = sorted((max(0.0,float(t)),Path(p)) for t,p in events if 0 <= t < duration)
    if not events:
        return video
    destination = output or video
    destination.parent.mkdir(parents=True,exist_ok=True)
    handle, name = tempfile.mkstemp(suffix='.mp4',prefix='.wildshield-audio-',dir=destination.parent)
    os.close(handle)
    temp = Path(name)
    command = [get_ffmpeg_exe(),'-nostdin','-y','-loglevel','error','-i',str(video)]
    filters=[]
    for i,(at,sound) in enumerate(events):
        if not sound.is_file():
            temp.unlink(missing_ok=True)
            raise ValueError(f'Alert sound not found: {sound}')
        command.extend(['-i',str(sound)])
        end=events[i+1][0] if i+1<len(events) else duration
        filters.append(f'[{i+1}:a]aresample=44100,aformat=channel_layouts=stereo,atrim=duration={max(.001,end-at):.6f},asetpts=PTS-STARTPTS,adelay={round(at*1000)}:all=1[a{i}]')
    labels=''.join(f'[a{i}]' for i in range(len(events)))
    filters.append(f'{labels}amix=inputs={len(events)}:normalize=0,apad,atrim=duration={duration:.6f}[audio]')
    command.extend(['-filter_complex',';'.join(filters),'-map','0:v:0','-map','[audio]','-c:v','copy','-c:a','aac','-b:a','192k','-t',str(duration),'-movflags','+faststart',str(temp)])
    try:
        result=subprocess.run(command,capture_output=True,text=True)
        if result.returncode:
            raise RuntimeError(result.stderr[-2000:])
        os.replace(temp,destination)
    finally:
        temp.unlink(missing_ok=True)
    return destination
