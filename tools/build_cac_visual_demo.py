"""Build a labeled review video from preserved person-trial evidence."""
from pathlib import Path
import json
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/competition'
W,H,FPS=1280,720,10
OUT.mkdir(parents=True,exist_ok=True)
writer=cv2.VideoWriter(str(OUT/'WildShield_CAC_visual_demo.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),FPS,(W,H))
assert writer.isOpened()

def text(frame,line,x,y,scale=.7,color=(230,235,240)):
    cv2.putText(frame,line,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,2,cv2.LINE_AA)

def card(title,lines,seconds):
    frame=np.full((H,W,3),(25,30,35),np.uint8)
    text(frame,'WildShield',60,85,1.5,(100,220,160))
    text(frame,title,60,165,1.0)
    for i,line in enumerate(lines): text(frame,line,60,245+i*57,.7)
    for _ in range(round(seconds*FPS)):writer.write(frame)

def fitted(img,width,height):
    scale=min(width/img.shape[1],height/img.shape[0])
    return cv2.resize(img,(round(img.shape[1]*scale),round(img.shape[0]*scale)))

def slide(img,title,footer,seconds):
    frame=np.full((H,W,3),(25,30,35),np.uint8)
    text(frame,title,30,42,.76)
    pic=fitted(img,W,H-150)
    x=(W-pic.shape[1])//2;y=65+(H-150-pic.shape[0])//2
    frame[y:y+pic.shape[0],x:x+pic.shape[1]]=pic
    text(frame,footer,25,H-35,.56)
    for _ in range(round(seconds*FPS)):writer.write(frame)

card('Protecting wildlife near our roads',[
    'Inspired by seeing deer beside roads around Georgetown, Texas.',
    'An educational RGB + thermal sensing prototype.',
    'Recorded experiments with real RGB and thermal sensor data.',
    'Goal: explore earlier warnings for wildlife near roads.'
],8)
manifest=[]
for lighting in ('dim','dark'):
    tid=f'fusion-{lighting}-present-mounted-03'
    snapshot=ROOT/'outputs/trials'/f'{tid}.png'
    img=cv2.imread(str(snapshot));assert img is not None
    h,w=img.shape[:2]
    # Broad upper target regions cover facial features in both sensor views.
    # The bottom status labels and RGB target rectangle remain visible.
    for x1,y1,x2,y2 in ((.07,.08,.40,.64),(.53,.09,.89,.69)):
        a,b,c,d=int(x1*w),int(y1*h),int(x2*w),int(y2*h)
        roi=img[b:d,a:c]
        img[b:d,a:c]=cv2.GaussianBlur(roi,(0,0),45)
        # Coarse mosaic reinforces anonymization after the blur.
        small=cv2.resize(img[b:d,a:c],(5,5),interpolation=cv2.INTER_AREA)
        img[b:d,a:c]=cv2.resize(small,(c-a,d-b),interpolation=cv2.INTER_NEAREST)
    rows=[json.loads(s) for s in (ROOT/'outputs/trials'/f'{tid}.jsonl').read_text().splitlines() if s.strip()]
    slide(img,f'{lighting.upper()} | Saved final dashboard screenshot | Face regions obscured',
          'Person target, approx. 3 m | RGB + thermal temporal confirmation | Radar is SIMULATED',7)
    # Replay measured thermal sequence without fabricating missing RGB frames.
    for i,row in enumerate(rows):
        temps=np.array(row['temperatures_c'],np.float32).reshape(24,32)
        lo,hi=float(temps.min()),float(temps.max())
        norm=np.uint8(np.clip((temps-lo)/max(hi-lo,.01)*255,0,255))
        heat=cv2.applyColorMap(norm,cv2.COLORMAP_INFERNO)
        # Obscure broad head zone in the thermal video before enlargement.
        heat[5:17,9:26]=cv2.resize(cv2.resize(heat[5:17,9:26],(2,2)),(17,12),interpolation=cv2.INTER_NEAREST)
        heat=cv2.resize(heat,(640,480),interpolation=cv2.INTER_NEAREST)
        frame=np.full((H,W,3),(25,30,35),np.uint8)
        text(frame,f'{lighting.upper()} | THERMAL LOG REPLAY - recorded sensor samples',25,45,.8)
        frame[100:580,25:665]=heat
        text(frame,'Head region obscured',700,140,.62)
        text(frame,f'Saved sample: {i+1}/{len(rows)}',700,205,.65)
        text(frame,'Thermal: '+('CONFIRMED' if row['thermal_confirmed'] else 'PENDING'),700,265,.65)
        text(frame,'RGB detections: '+str(len(row.get('rgb_detections',[]))),700,325,.65)
        text(frame,f'Temperature range: {lo:.1f}-{hi:.1f} C',700,385,.55)
        text(frame,'No RGB video was saved.',700,460,.6)
        text(frame,'One saved sample per second;',700,510,.55)
        text(frame,'timing normalized for this replay.',700,545,.55)
        text(frame,'Measured thermal data | 32 x 24 sensor | No radar values used in this replay',25,650,.64)
        for _ in range(FPS):writer.write(frame)
    manifest.append({'trial':tid,'snapshot':str(snapshot.relative_to(ROOT)),'samples':len(rows)})
img=cv2.imread(str(ROOT/'outputs/trials/deer_image_test_annotated.jpg'));assert img is not None
slide(img,'DEER RECOGNITION | Previously processed still image',
      'RGB-only YOLO-World test | No thermal sensing of this deer | Not live wildlife footage',9)
card('What these tests demonstrate',[
    'Person tests: RGB and thermal signals combined in the same cycle.',
    'Deer image: a separate visual recognition example.',
    'Sensors are not spatially calibrated; no live-deer fusion test.',
    'Simulated radar does not establish collision prediction.',
    'Educational prototype; not for driving decisions.'
],9)
concept=cv2.imread(str(ROOT/'WildShield-product-concept.png'));assert concept is not None
slide(concept,'FUTURE BUILD-OUT | Existing hardware integration or separate device',
      'Planned options: integrate into existing hardware, or offer a separate sensor device',7)
card('WildShield | Building toward wildlife protection',[
    'Python, OpenCV, Ultralytics YOLO / YOLO-World, MLX90640.',
    'Existing evidence preserved; faces obscured in this export.',
    'AI-assisted video assembly and concept image: OpenAI Codex.',
    'Next: spatial calibration and live-wildlife validation.',
    'Separate deer-image recognition test; no deer thermal capture.'
],7)
writer.release()
cap=cv2.VideoCapture(str(OUT/'WildShield_CAC_visual_demo.mp4'))
frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
ok,frame=cap.read();assert ok
cv2.imwrite(str(OUT/'WildShield_CAC_visual_preview.jpg'),frame)
cap.release()
(OUT/'WildShield_CAC_visual_manifest.json').write_text(json.dumps({'sources':manifest,'duration_seconds':frames/fps,'narration_ready_visual_cut':True,'deer_source':'outputs/trials/deer_image_test_annotated.jpg','privacy':'Broad RGB and thermal head regions blurred/mosaicked; originals unchanged'},indent=2))
print(f'Created {frames/fps:.1f}-second review video; {frames} frames, {fps} fps.')
