"""Create a labeled copy with risk-alert audio reconstructed from a trial log."""
import argparse,json
from pathlib import Path
import cv2
from wildshield.io.audio_video import mux_alert_audio
from wildshield.ui.alert import RiskAudioAlert


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video',type=Path)
    args=parser.parse_args()
    rows=[json.loads(line) for line in args.video.with_suffix('.jsonl').read_text().splitlines() if line.strip()]
    cap=cv2.VideoCapture(str(args.video));fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
    if fps<=0 or count<=0:raise SystemExit('Cannot read video frame rate/count')
    if count!=len(rows):raise SystemExit('Frame/log count mismatch; cannot align risk audio reliably')
    sounds=Path(__file__).resolve().parents[1]/'assets/sounds'
    events=[]
    def record(message):
        sound = 'risk_danger.aiff' if message.startswith('DANGER') else 'risk_warning.aiff'
        events.append((i/fps, sounds/sound))
    gate=RiskAudioAlert(record)
    for i,row in enumerate(rows):
        gate.update((row.get('risk') or {}).get('level'))
    if not events:raise SystemExit('No WARNING/DANGER transitions to reconstruct')
    output=args.video.with_name(args.video.stem+'_with_alerts.mp4')
    if output.exists():raise SystemExit(f'Refusing to overwrite {output}')
    mux_alert_audio(args.video,events,count/fps,output)
    output.with_suffix('.audio.json').write_text(json.dumps({'source_video':str(args.video),'source_log':str(args.video.with_suffix('.jsonl')),'audio':'Risk-state alert sounds reconstructed from saved log; not live microphone capture','radar':'Simulated','events':[{'video_seconds':t,'sound':str(p)} for t,p in events]},indent=2)+'\n')
    print(f'Saved {output}; {len(events)} reconstructed risk alerts; original unchanged.')
if __name__=='__main__':main()
