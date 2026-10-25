"""Slide-ready charts from saved trial metrics; no pooled target groups."""
from pathlib import Path
import csv,statistics,os
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/comparison_report/slide_charts';OUT.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/wildshield-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.titleweight':'bold','svg.fonttype':'none','savefig.facecolor':'white'})
colors=['#2677B8','#E98926','#6C529B'];lights=['bright','dim','dark']
def read(p):
 with (ROOT/p).open() as f:return list(csv.DictReader(f))
person=read('outputs/comparison_report/sensor_performance.csv')
animal=read('outputs/competition/animal_video_results.csv')
replay=read('outputs/competition/adaptive_fusion_replay.csv')
export=[]
def base(title,subtitle):
 fig,ax=plt.subplots(figsize=(16,9));fig.subplots_adjust(left=.10,right=.97,top=.76,bottom=.23)
 fig.text(.07,.93,title,fontsize=29,weight='bold',color='#152B3C')
 fig.text(.07,.865,subtitle,fontsize=16,color='#4D6170')
 ax.set_ylim(0,120);ax.set_yticks([0,25,50,75,100]);ax.set_yticklabels(['0%','25%','50%','75%','100%']);ax.grid(axis='y',color='#E4E9ED');ax.set_axisbelow(True);ax.tick_params(length=0,pad=10)
 ax.set_xticks(range(3),['Bright','Dim','Dark']);ax.set_ylabel('Mean detected frames per trial',labelpad=15)
 return fig,ax
def save(fig,name,footer):
 fig.text(.07,.12,footer,fontsize=13,color='#465A69',linespacing=1.6)
 fig.text(.07,.04,'WildShield | Controlled prototype evidence',fontsize=12,color='#71808B')
 for ext in ['png','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=180)
 plt.close(fig)
def groups(rows,field,light):return [100*float(r[field]) for r in rows if r.get('condition',r.get('lighting'))==light and r['target_present']=='True']
def bars(rows,fields,title,subtitle,name,footer):
 fig,ax=base(title,subtitle);width=.23
 for j,(field,label) in enumerate(fields):
  xs=np.arange(3)+(j-(len(fields)-1)/2)*width
  vals=[groups(rows,field,l) for l in lights];means=[statistics.mean(v) for v in vals];sds=[statistics.stdev(v) for v in vals]
  ax.bar(xs,means,width*.9,color=colors[j],label=label,zorder=2)
  ax.errorbar(xs,means,yerr=sds,fmt='none',ecolor='#1C3446',capsize=5,lw=1.4,zorder=4)
  for x,m,sd,v,light in zip(xs,means,sds,vals,lights):
   ax.scatter(x+np.linspace(-.035,.035,len(v)),v,s=28,color='white',edgecolor='#1C3446',linewidth=.8,zorder=5)
   ax.text(x,m+3,f'{m:.1f}%',ha='center',fontsize=13,weight='bold',color='#152B3C',bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=1),zorder=6)
   export.append(dict(chart=name,series=label,lighting=light,n=len(v),mean_percent=m,sample_sd_pp=sd))
 ax.legend(loc='upper left',bbox_to_anchor=(0,1.20),ncol=len(fields),frameon=False,fontsize=15)
 save(fig,name,footer)
bars(person,[('rgb_detection_rate','RGB'),('thermal_detection_rate','Thermal'),('fusion_detection_rate','RGB + thermal')],
 'Thermal confirmation stayed consistent across lighting',
 'PERSON TARGET | Approximately 3 m | Three target-present trials per lighting condition',
 '01_person_sensor_comparison',
 'Dots: individual trials. Whiskers: ±1 sample SD (n=3). Thermal confirmation requires two consecutive frames.\nOriginal AND fusion; temporal confirmation only. Small controlled sample; mounting varied in earlier trials.')
bars(animal,[('rgb_rate','RGB'),('thermal_rate','Thermal'),('fusion_rate','RGB + thermal')],
 'RGB and Thermal Detection Across Lighting Conditions',
 'Warmed animal-model tests | Three trials per condition | Approximately 1 m',
 '02_animal_model_sensor_comparison',
 'Dots: individual trials. Whiskers: ±1 sample SD (n=3). Target: warmed stuffed-animal surrogate, not wildlife.\nOriginal AND fusion. Lighting categories were observational; radar excluded from measured detection results.')
bars(replay,[('legacy_rate','Original AND rule'),('adaptive_rate','Adaptive actionable decision')],
 'Adaptive rules changed warnings in the software replay',
 'SOFTWARE REPLAY | Same animal-model trials | Approaching-radar observations were SIMULATED',
 '03_adaptive_replay_simulated',
 'Dots: trials. Whiskers: ±1 sample SD. Actionable = confirmed or warning; differs from original confirmation.\nRetrospective replay, not new physical trials. This does not validate real radar or collision prediction.')
# All negative trials shown, preserving the distinction between target groups.
fig,ax=plt.subplots(figsize=(16,9));fig.subplots_adjust(left=.23,right=.97,top=.76,bottom=.25)
fig.text(.07,.93,'Accepted empty controls produced no detections',fontsize=29,weight='bold',color='#152B3C')
fig.text(.07,.865,'18 ACCEPTED NEGATIVE TRIALS | Nine person controls + nine animal-model controls',fontsize=16,color='#4D6170')
for group,y,rows,field in [('Person setup',1,person,'fusion_detection_rate'),('Animal-model setup',0,animal,'fusion_rate')]:
 neg=sorted([r for r in rows if r['target_present']=='False'],key=lambda r:(lights.index(r.get('condition',r.get('lighting'))),r['trial_id']));assert len(neg)==9
 for i,r in enumerate(neg):
  assert all(float(r[k])==0 for k in (['rgb_detection_rate','thermal_detection_rate','fusion_detection_rate'] if group=='Person setup' else ['rgb_rate','thermal_rate','fusion_rate']))
  ax.scatter(i,y,s=800,color='#218875',marker='s');ax.text(i,y,'0',ha='center',va='center',color='white',weight='bold',fontsize=20)
ax.set_xlim(-.7,8.7);ax.set_ylim(-.6,1.6);ax.set_xticks([1,4,7],['Bright: 3 trials','Dim: 3 trials','Dark: 3 trials']);ax.set_yticks([0,1],['Animal-model setup','Person setup']);ax.tick_params(length=0,pad=18);ax.spines['bottom'].set_visible(False)
save(fig,'04_empty_controls','Each square is one trial with zero confirmed RGB, thermal, and original-fusion detections.\nExcluded controls remain documented in source manifests. These short tests do not establish a real-world false-alarm rate.')
with (OUT/'chart_data.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(export[0]));w.writeheader();w.writerows(export)
(OUT/'README.md').write_text('''# WildShield slide charts

Four 16:9 charts, each in high-resolution PNG and editable SVG. PNGs are 2880 × 1620 pixels. SVG labels remain text.

1. Person sensor comparison: outputs/comparison_report/sensor_performance.csv.
2. Warmed animal-model comparison: outputs/competition/animal_video_results.csv.
3. Adaptive replay: outputs/competition/adaptive_fusion_replay.csv. Simulated radar; different output definition from original confirmation.
4. Empty controls: all nine accepted negative trials per physical target group; checks RGB, thermal and original fusion.

Means weight each trial equally. Error bars are sample standard deviations across three trials, not confidence intervals. Dots show each trial. All measured detection data and original source files remain unchanged. No results pooled between target types. Accepted controls only; exclusions are in original manifests.

Chart 1: some earlier trials were not fixed-mounted. Chart 2: warmed surrogate, not live wildlife. Chart 3: retrospective simulation-supported software behavior, not validated radar performance. No overall wildlife accuracy or collision-prevention claim is supported.

chart_data.csv contains the plotted means and sample standard deviations. Charts generated with Python/Matplotlib with OpenAI Codex assistance.
''')
print('Created four PNG/SVG chart pairs and chart_data.csv in '+str(OUT))
