"""Generate an editable 8-slide deck and matching PDF layout preview.

Requires python-pptx and reportlab. Pass the approved script path and output
directory. Speaker notes retain the narration and recording instructions.
"""
import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor

ROOT = Path(sys.argv[2])
ROOT.mkdir(parents=True, exist_ok=True)
script = Path(sys.argv[1]).read_text()
rows = [r.split('|')[1:-1] for r in script.splitlines() if r.startswith('| **')]
assert len(rows) == 10
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333333), Inches(7.5)
prs.core_properties.title = 'Ranking-Aware Concept-Level Sparse Retrieval'
prs.core_properties.subject = 'AI6103 five-minute project presentation'
prs.core_properties.author = 'SAE Retrieval Project Team'
pdf = canvas.Canvas(str(ROOT / 'SAE_Retrieval_Video.pdf'), pagesize=(960, 540))
pdf.setTitle('SAE Retrieval — video presentation preview')
for font, filename in [('Arial', 'Arial.ttf'), ('Arial-Bold', 'Arial Bold.ttf')]:
    path = Path('/System/Library/Fonts/Supplemental') / filename
    pdfmetrics.registerFont(TTFont(font, str(path)))

BG='101827'; PANEL='1B283C'; WHITE='F4F7FC'; MUTED='B0BED0'; CYAN='50DACB'; ORANGE='FFB36C'; LINE='33445C'
slide = None
page = 0

def rgb(c): return RGBColor.from_string(c)
def unit(x): return Inches(x / 96)

def rect(x,y,w,h,color=PANEL,radius=12):
    shape=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,unit(x),unit(y),unit(w),unit(h))
    shape.fill.solid();shape.fill.fore_color.rgb=rgb(color);shape.line.fill.background()
    if radius:
        shape.adjustments[0]=min(radius/min(w,h),.5)
    pdf.setFillColor(HexColor('#'+color))
    pdf.roundRect(x*.75,(720-y-h)*.75,w*.75,h*.75,radius*.75,stroke=0,fill=1)

def txt(x,y,w,text,size=24,color=WHITE,bold=False,h=None):
    lines=text.split('\n'); h=h or (len(lines)*size*1.3+10)
    box=slide.shapes.add_textbox(unit(x),unit(y),unit(w),unit(h));tf=box.text_frame
    tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0
    tf.word_wrap=False
    for i,line in enumerate(lines):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
        p.text=line;p.font.name='Arial';p.font.size=Pt(size*.75);p.font.bold=bold;p.font.color.rgb=rgb(color)
        p.space_after=Pt(size*.75*.26)
        pdf.setFillColor(HexColor('#'+color));pdf.setFont('Arial-Bold' if bold else 'Arial',size*.75)
        pdf.drawString(x*.75,(720-y-size*.94-i*size*1.3)*.75,line)
        assert pdfmetrics.stringWidth(line,'Arial-Bold' if bold else 'Arial',size) <= w+3, (page,line,'text too wide')
    assert y+h<=720, (page,text,'text below slide')

def new(title,kicker,time,note_indices):
    global slide,page
    if page:pdf.showPage()
    page+=1;slide=prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid();slide.background.fill.fore_color.rgb=rgb(BG)
    pdf.setFillColor(HexColor('#'+BG));pdf.rect(0,0,960,540,fill=1,stroke=0)
    txt(56,30,1000,kicker.upper(),16,CYAN,True)
    txt(56,67,1180,title,40,WHITE,True)
    rect(56,652,1168,2,LINE,0)
    txt(56,674,880,'AI6103  /  SAE RETRIEVAL',14,MUTED)
    txt(1020,674,210,f'{time}   /   {page:02}',14,MUTED)
    notes=[]
    for n in note_indices:
        stamp,screen,speech=rows[n]
        notes.append(stamp.replace('**','').strip()+'\n\nSHOW: '+screen.strip().replace('**','')+'\n\nSAY: '+speech.strip().strip('“”'))
    slide.notes_slide.notes_text_frame.text='\n\n---\n\n'.join(notes)

def card(x,y,w,h,title,body,accent=CYAN):
    rect(x,y,w,h);txt(x+24,y+22,w-48,title,27,accent,True)
    txt(x+24,y+68,w-48,body,22,MUTED)

def bars(items,x,y,w,labelw=230,maxval=.4,gap=78):
    for i,(label,value,display,color) in enumerate(items):
        yy=y+i*gap;txt(x,yy+6,labelw,label,23,WHITE)
        rect(x+labelw,yy+4,w-labelw-170,36,LINE,0)
        rect(x+labelw,yy+4,(w-labelw-170)*value/maxval,36,color,0)
        txt(x+w-150,yy+4,170,display,24,color,True)

# 01
new('Teaching sparse concepts to rank','Project overview','0:00–0:25',[0])
txt(56,137,1150,'Ranking-Aware Concept-Level Sparse Retrieval with Sparse Autoencoders',24,MUTED)
stages=[('QUESTION','Text input'),('SIMLM','Dense vector'),('SAE','Sparse concepts'),('INDEX','Posting lists'),('RESULTS','Ranked passages')]
for i,(a,b) in enumerate(stages):
    x=56+i*236;rect(x,225,218,125);txt(x+18,250,185,a,24,CYAN,True);txt(x+18,293,190,b,20)
    if i<4:txt(x+220,268,20,'›',26,MUTED)
txt(56,391,1150,'Can ranking supervision improve quality without hiding its cost?',30,WHITE,True)
txt(56,474,1150,'Vishakha Maheshwari  ·  Riddhima Jain  ·  Vishakha Kulkarni',22,MUTED)
txt(56,511,1150,'Mohit Sharma  ·  Kush Hemal Maniar',22,MUTED)
txt(56,579,1150,'8.84 million passages  •  One frozen starting baseline  •  Five interventions',20,CYAN)

# 02
new('Five stages. Controlled comparisons.','Study design','0:25–0:45',[1])
names=[('P1','Scoring','Learn concept impacts'),('P2','SAE training','Add ranking supervision'),('P3','Passage encoding','Vary concept count'),('P4','Concept vocabulary','Prune costly postings'),('P5','Query encoding','Expand query concepts')]
for i,(tag,stage,goal) in enumerate(names):
    yy=151+i*71;rect(56,yy,1168,59);txt(76,yy+14,65,tag,24,CYAN,True);txt(161,yy+14,330,stage,24,WHITE,True);txt(560,yy+14,630,goal,23,MUTED)
txt(56,541,1168,'SELECT  /  5,000 validation queries; no dev or TREC tuning',24,ORANGE,True)
txt(56,588,1168,'TEST  /  6,980 dev queries + TREC DL19 / DL20; quality and cost',23,MUTED)

# 03
new('Learned scoring gives the largest standalone gain','P1 · Vishakha Maheshwari','0:45–1:40',[2,3])
txt(56,137,1130,'Shared concept features → 193-parameter MLP → Sum of learned impacts',23,MUTED)
txt(56,196,800,'FULL-CORPUS DEV MRR@10',17,CYAN,True)
bars([('Released BM25',.3157805,'0.3158',MUTED),('Tuned BM25',.3239190,'0.3239',ORANGE),('Learned scoring',.3600281,'0.3600',CYAN)],56,235,870,maxval=.4)
rect(972,205,252,260);txt(996,233,210,'+11.1%',48,CYAN,True);txt(996,303,210,'relative to\ntuned BM25',23,MUTED);txt(996,389,210,'±0.0015 SD\nover 3 seeds',21)
card(56,500,565,122,'Same passage postings','Preselected seed gain: significant')
card(643,500,581,122,'Trade-offs remain','122 → 192 ms on A40; TREC recall falls',ORANGE)

# 04
new('Train the SAE for retrieval, not only reconstruction','P2 · Riddhima Jain','1:40–2:10',[4])
card(56,158,560,145,'Reconstruction loss','Preserve the embedding signal')
card(640,158,584,145,'Ranking-distillation loss','Match SimLM through the CL-SR score')
txt(56,336,1150,'DEV MRR@10  /  SELECTED SEED-42 RUN',17,CYAN,True)
bars([('Baseline B0',.316,'≈0.316',MUTED),('Training controls',.315,'≈0.315',ORANGE),('Ranking-aware SAE',.33027,'≈0.330',CYAN)],56,377,1168,maxval=.4,gap=66)
txt(56,596,1168,'Reconstruction degradation remains inside the predefined 10% budget.',23,MUTED)

# 05
new('Adaptive sparsity misses the quality–storage target','P3 · Vishakha Kulkarni','2:10–2:40',[5])
txt(56,142,1168,'Retain concepts above a threshold, subject to minimum / maximum counts.',24,MUTED)
txt(56,203,1168,'FULL-CORPUS VALIDATION  /  NOT A STANDALONE DEV RESULT',18,ORANGE,True)
headers=[('Setting',80),('MRR@10',575),('Concepts / passage',790),('Index',1090)]
for label,x in headers:txt(x,263,400 if x<500 else 280,label,22,MUTED,True)
for yy,label,mrr,concepts,size in [(308,'Fixed k = 32','0.3427','32.0','2.64 GiB'),(392,'Adaptive','0.3335','32.6','2.69 GiB')]:
    rect(56,yy,1168,67);txt(80,yy+18,470,label,27,WHITE,True);txt(575,yy+18,180,mrr,27,CYAN if yy==308 else ORANGE,True);txt(790,yy+18,220,concepts,27);txt(1070,yy+20,150,size,24)
card(56,510,1168,115,'Allowed MRR drop ≤ 0.005; observed drop ≈ 0.0092','Negative standalone result; adaptive encoding also appears in combined runs.',ORANGE)

# 06
new('Learn which postings to remove','P4 · Mohit Sharma','2:40–3:15',[6])
txt(56,139,1168,'Distillation preserves ranking; a learned gate penalizes costly concepts.',24,MUTED)
rect(56,201,338,198);txt(80,231,295,'23.5%',54,CYAN,True);txt(80,309,295,'fewer postings',25)
rect(418,201,806,198);txt(445,231,750,'0.655 → 0.556 GB',44,CYAN,True);txt(445,306,750,'Measured Lucene footprint  /  seed 0',24,MUTED);txt(445,350,750,'Approximately 15% smaller index',23)
txt(56,439,1168,'DEV MRR@10 LOSS AT THE SAME POSTING REDUCTION',18,MUTED,True)
bars([('Learned gate',.001810,'−0.0018',CYAN),('Frequency pruning',.015627,'−0.0156',ORANGE)],56,482,1168,maxval=.02,gap=65)
txt(56,615,1168,'Gate MRR loss: not significant; this is not proof of equivalence.',20,MUTED)

# 07
new('Query expansion does not beat the simple control','P5 · Kush Hemal Maniar','3:15–3:50',[7])
txt(56,140,1168,'Near-miss concepts + decoder neighbours → Expansion MLP → Expanded query',23,MUTED)
txt(56,209,1168,'DEV RECALL@1000  /  FIRST-SEED COMPARISON',18,CYAN,True)
bars([('Baseline B0',.9472,'0.9472',MUTED),('Learned expansion',.9488,'0.9488',ORANGE),('Lower threshold',.9522,'0.9522',CYAN)],56,261,1168,maxval=1,gap=82)
card(56,535,1168,92,'Small recall change; no significant learned gain after correction','Many missed passages already share concepts: ranking remains a bottleneck.',ORANGE)

# 08
new('Combining stages adds limited benefit in one seed','Combined evidence & conclusion','3:50–5:00',[8,9])
txt(56,139,1168,'DEV MRR@10  /  MATCHED SEED 42',18,CYAN,True)
for i,(label,value) in enumerate([('Standalone learned scoring','0.3588'),('Ranking-aware SAE + learned scoring','0.3608'),('All five stages','0.3608')]):
    yy=183+i*65;rect(56,yy,1168,54);txt(78,yy+12,900,label,25);txt(1070,yy+12,150,value,27,CYAN,True)
txt(56,398,1168,'Strongest quality gain: scoring.  Storage reduction: learned pruning.',26,WHITE,True)
txt(56,449,1168,'Measure quality, recall, storage and latency together.',30,CYAN,True)
txt(56,511,1168,'Limits: one dataset family; combined evidence is single-seed.',22,MUTED)
txt(56,548,1168,'No established significant combined gain; compare cost within matched backends.',22,MUTED)
txt(56,603,1168,'github.com/Vishakha-Kulkarni/sae-retrieval',22,ORANGE)

sources = [
    'Base paper: https://aclanthology.org/2025.emnlp-main.1345/ ; approved VIDEO_SCRIPT_TEMP.md',
    'Shared split: splits/validation.qids; method-specific protocols in the project report.',
    'experiments/p1/full_corpus_heldout.json; experiments/p1/vocabulary_mismatch.json',
    'HF: RiddhimaJ/sae-simlm-k32-baseline, revision 328c5f956177d74a50ad31d9b4983a184e831362, runs/clsr_fast_max24/listwise_bm25_w0.2_s42/dev.metrics.txt; controls as reported by P2.',
    'docs/P3_results.md and sae-simlm-k32-baseline/data_titled/p3_full_eval/*validation*.json',
    'p4/results/significance/gate_vs_full_lam0.4.json; p4/results/significance/gate_vs_df_lam0.4.json; p4/results/val_atom/meta/gate_lam0.4.meta.json',
    'docs/P5_method_note.md: final first-seed dev comparisons; no claim of independent raw-run verification.',
    'Combined values: teammate terminal screenshot supplied in chat; aggregates transcribed, not independently recomputed. Standalone P1 seed42: experiments/p1/full_corpus_heldout.json. Combined seed42 label follows the team report; screenshot alone does not establish the seed.'
]
for sl, source in zip(prs.slides,sources):
    sl.notes_slide.notes_text_frame.text += '\n\nSOURCES / EVIDENCE: '+source
prs.save(ROOT/'SAE_Retrieval_Video.pptx');pdf.save()
assert len(prs.slides)==8
print('Created editable PowerPoint and 8-page PDF preview:',ROOT)
