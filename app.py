import math, sys, time, cv2, numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication,QMainWindow,QWidget,QLabel,QPushButton,QComboBox,QSlider,QVBoxLayout,QHBoxLayout,QGridLayout,QFrame,QFileDialog,QMessageBox,QSizePolicy
from step2_camera import SimulatedCameraSource, VideoFileSource, VIEWPORT_WIDTH
from step4_search import SpiralScanner
from logger import PerformanceLogger
from core import RobustBeaconDetector,KalmanBeaconTracker2D,PIDController2D,SETPOINT_X,SETPOINT_Y,PIXELS_PER_DEGREE
FPS=30
STYLE="""
QMainWindow,QWidget{background:#08111f;color:#e5edf7;font-family:'Segoe UI';}
QFrame#card{background:#101c2d;border:1px solid #20344e;border-radius:14px;}
QLabel#title{color:#63d9ff;font-size:20px;font-weight:700;} QLabel#subtitle{color:#7186a1;font-size:11px;}
QLabel#section{color:#7ddcff;font-size:12px;font-weight:700;} QLabel#metricName{color:#7d91aa;font-size:10px;} QLabel#metricValue{color:#f3f7fb;font-size:17px;font-weight:700;}
QLabel#badge{background:#13283c;border:1px solid #245b79;border-radius:9px;padding:5px 10px;color:#72ddff;font-weight:700;}
QPushButton{background:#17283c;border:1px solid #2c4966;border-radius:8px;padding:8px 12px;color:#e8f2fc;font-weight:600;} QPushButton:hover{background:#203850;border-color:#4b89ad;}
QPushButton#primary{background:#087fa8;border-color:#0fa6d6;} QPushButton#success{background:#146c50;border-color:#24a875;}
QComboBox{background:#0c1828;border:1px solid #29425d;border-radius:7px;padding:7px;} QComboBox QAbstractItemView{background:#0d1a2a;color:#fff;selection-background-color:#164e68;}
QSlider::groove:horizontal{height:5px;background:#20344a;border-radius:2px;} QSlider::handle:horizontal{width:16px;margin:-6px 0;background:#53d4ff;border:2px solid #a6edff;border-radius:8px;}
QLabel#log{color:#91a7bf;background:#08111c;border-radius:8px;padding:7px;}
"""
class MetricCard(QFrame):
 def __init__(self,name,value='—'):
  super().__init__();self.setObjectName('card');l=QVBoxLayout(self);l.setContentsMargins(12,9,12,9);n=QLabel(name.upper());n.setObjectName('metricName');self.value=QLabel(value);self.value.setObjectName('metricValue');l.addWidget(n);l.addWidget(self.value)
 def set(self,text,color=None):
  self.value.setText(text)
  if color:self.value.setStyleSheet(f'color:{color};')
class SliderRow(QWidget):
 changed=Signal()
 def __init__(self,name,lo,hi,value,decimals=0):
  super().__init__();self.lo,self.hi,self.decimals=lo,hi,decimals;l=QVBoxLayout(self);l.setContentsMargins(0,3,0,3);top=QHBoxLayout();n=QLabel(name);n.setObjectName('metricName');self.val=QLabel();top.addWidget(n);top.addStretch();top.addWidget(self.val);l.addLayout(top);self.slider=QSlider(Qt.Horizontal);self.slider.setRange(0,1000);self.set_value(value);self.slider.valueChanged.connect(self.changed.emit);self.slider.valueChanged.connect(lambda:self.val.setText(self.display()));l.addWidget(self.slider)
 def value(self):return self.lo+(self.hi-self.lo)*self.slider.value()/1000
 def set_value(self,v):self.slider.setValue(round((float(v)-self.lo)/(self.hi-self.lo)*1000));self.val.setText(self.display())
 def display(self):return f'{self.value():.{self.decimals}f}'
class FSOCWindow(QMainWindow):
 def __init__(self):
  super().__init__();self.setWindowTitle('FSOC Command Center — Coarse Alignment');self.resize(1560,980);self.setMinimumSize(1180,760);self.source=SimulatedCameraSource(target_start=(450,1000),motion_type='horizontal');self.is_video_source=False;self.detector=RobustBeaconDetector();self.kf=KalmanBeaconTracker2D();self.pid=PIDController2D();self.logger=PerformanceLogger();self.pan_speed_deg=8.;self.fov_deg=4.;self.scanner=self.make_scanner();self.running=True;self.state='SEARCHING';self.frame_id=0;self.search_frames=0;self.frames_lost=0;self.acquisition_time=None;self.recent_errors=[];self.fps=30.;self.last_time=time.time();self.build_ui();self.timer=QTimer(self);self.timer.timeout.connect(self.update_loop);self.timer.start(33)
 def make_scanner(self):return SpiralScanner(self.source.scene_width/2,self.source.scene_height/2,(self.pan_speed_deg/FPS)*(VIEWPORT_WIDTH/self.fov_deg))
 def card(self,title):
  f=QFrame();f.setObjectName('card');l=QVBoxLayout(f);l.setContentsMargins(12,10,12,10);h=QLabel(title.upper());h.setObjectName('section');l.addWidget(h);return f,l
 def build_ui(self):
  root=QWidget();o=QVBoxLayout(root);o.setContentsMargins(14,12,14,12);o.setSpacing(10);self.setCentralWidget(root);head=QHBoxLayout();t=QVBoxLayout();a=QLabel('FSOC COMMAND CENTER');a.setObjectName('title');b=QLabel('AI-BASED VIRTUAL CAMERA TRACKING  •  COARSE ALIGNMENT  •  30 FPS');b.setObjectName('subtitle');t.addWidget(a);t.addWidget(b);head.addLayout(t);head.addStretch();self.status=QLabel('● SYSTEM ONLINE');self.status.setObjectName('badge');head.addWidget(self.status);o.addLayout(head)
  ds=QHBoxLayout();cam,l=self.card('Camera Viewport  •  640 × 480 FPA');self.cam=QLabel();self.cam.setAlignment(Qt.AlignCenter);self.cam.setMinimumSize(520,390);self.cam.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding);self.cam.setStyleSheet('background:#02060b;border-radius:9px;');l.addWidget(self.cam);ds.addWidget(cam,1);world,l=self.card('World Overview  •  2000 × 2000');self.world=QLabel();self.world.setAlignment(Qt.AlignCenter);self.world.setMinimumSize(520,390);self.world.setStyleSheet('background:#02060b;border-radius:9px;');l.addWidget(self.world);ds.addWidget(world,1);o.addLayout(ds,1)
  bot=QHBoxLayout();o.addLayout(bot);tele,l=self.card('Live Telemetry');g=QGridLayout();self.m_state=MetricCard('Operating State','SEARCHING');self.m_err=MetricCard('Tracking Error');self.m_acq=MetricCard('Acquisition');self.m_rmse=MetricCard('Centroid RMSE');self.m_lock=MetricCard('Lock Retention');self.m_perf=MetricCard('FPS / Latency');
  for i,c in enumerate([self.m_state,self.m_err,self.m_acq,self.m_rmse,self.m_lock,self.m_perf]):g.addWidget(c,i//2,i%2)
  l.addLayout(g);bot.addWidget(tele,1)
  ctrl,l=self.card('Optical Environment & PTZ');self.motion=QComboBox();self.motion.addItems(['horizontal','straight_line','circular','figure_8','random']);self.motion.currentTextChanged.connect(self.on_motion);l.addWidget(self.row('Target Motion',self.motion));self.noise=QComboBox();self.noise.addItems(['none','salt_pepper','gaussian','poisson']);self.noise.currentTextChanged.connect(self.on_noise);l.addWidget(self.row('Noise Mode',self.noise));self.noise_lvl=SliderRow('Noise Level / Std',0,20,10,1);self.noise_lvl.changed.connect(self.on_noise);l.addWidget(self.noise_lvl);self.atmos=QComboBox();self.atmos.addItems(['clear','haze','fog','rain','low_light']);self.atmos.currentTextChanged.connect(self.on_atmos);l.addWidget(self.row('Atmosphere',self.atmos));self.atmos_sev=SliderRow('Atmosphere Severity',0,100,75);self.atmos_sev.changed.connect(self.on_atmos);l.addWidget(self.atmos_sev);self.jitter=SliderRow('Camera Jitter ±px',0,20,0,1);self.jitter.changed.connect(lambda:self.setattr_source('jitter_px',self.jitter.value()));l.addWidget(self.jitter);self.drift=SliderRow('Platform Drift ±px',0,20,0,1);self.drift.changed.connect(lambda:self.setattr_source('platform_drift_px',self.drift.value()));l.addWidget(self.drift);self.speed=SliderRow('Pan / Tilt Speed °/s',5,10,8,1);self.speed.changed.connect(self.on_speed);l.addWidget(self.speed);bot.addWidget(ctrl,1)
  act,l=self.card('Presets & Actions');g=QGridLayout();
  for i,(txt,key) in enumerate([('CLEAR SKY','clear'),('10% S&P NOISE','sp'),('HEAVY FOG 75%','fog'),('HIGH JITTER','jit')]):
   q=QPushButton(txt);q.clicked.connect(lambda _,k=key:self.preset(k));g.addWidget(q,i//2,i%2)
  l.addLayout(g);r=QHBoxLayout();self.pause=QPushButton('PAUSE');self.pause.clicked.connect(self.toggle_pause);q=QPushButton('RESET');q.clicked.connect(self.reset_sim);v=QPushButton('LOAD BP-2 MP4');v.setObjectName('primary');v.clicked.connect(self.load_video);r.addWidget(self.pause);r.addWidget(q);r.addWidget(v);l.addLayout(r);e=QPushButton('EXPORT BENCHMARK REPORT  •  CSV / JSON');e.setObjectName('success');e.clicked.connect(self.export_report);l.addWidget(e);self.log=QLabel('READY  • Simulation initialized');self.log.setObjectName('log');l.addWidget(self.log);bot.addWidget(act,1)
 def row(self,n,w):
  x=QWidget();h=QHBoxLayout(x);h.setContentsMargins(0,2,0,2);a=QLabel(n);a.setObjectName('metricName');h.addWidget(a);h.addStretch();h.addWidget(w);return x
 def setattr_source(self,n,v):setattr(self.source,n,v)
 def on_motion(self,v):
  if hasattr(self,'source'):self.source.set_motion_type(v);self.pid.reset();self.kf.reset();self.state='SEARCHING';self.search_frames=0;self.acquisition_time=None;self.recent_errors.clear()
 def on_noise(self):self.source.noise_mode=self.noise.currentText();self.source.noise_level=self.noise_lvl.value()
 def on_atmos(self):self.source.atmosphere_mode=self.atmos.currentText();self.source.atmosphere_severity=self.atmos_sev.value()/100
 def on_speed(self):self.pan_speed_deg=self.speed.value();self.scanner.update_speed((self.pan_speed_deg/FPS)*(VIEWPORT_WIDTH/self.fov_deg))
 def preset(self,k):
  if k=='clear':self.atmos.setCurrentText('clear');self.atmos_sev.set_value(0);self.noise.setCurrentText('none');self.noise_lvl.set_value(0);self.jitter.set_value(0);self.drift.set_value(0)
  elif k=='sp':self.atmos.setCurrentText('clear');self.noise.setCurrentText('salt_pepper');self.noise_lvl.set_value(10);self.jitter.set_value(0);self.drift.set_value(0)
  elif k=='fog':self.atmos.setCurrentText('fog');self.atmos_sev.set_value(75);self.noise.setCurrentText('none');self.noise_lvl.set_value(0);self.jitter.set_value(0);self.drift.set_value(0)
  else:self.atmos.setCurrentText('clear');self.noise.setCurrentText('none');self.jitter.set_value(20);self.drift.set_value(15)
  self.on_atmos();self.on_noise()
 def toggle_pause(self):self.running=not self.running;self.pause.setText('RESUME' if not self.running else 'PAUSE');self.log.setText(('PAUSED' if not self.running else 'RUNNING')+'  • Simulation control')
 def reset_sim(self):
  self.source=SimulatedCameraSource(target_start=(450,1000),motion_type=self.motion.currentText());self.on_noise();self.on_atmos();self.source.jitter_px=self.jitter.value();self.source.platform_drift_px=self.drift.value();self.on_speed();self.pid.reset();self.kf.reset();self.scanner=self.make_scanner();self.state='SEARCHING';self.search_frames=0;self.acquisition_time=None;self.recent_errors.clear();self.is_video_source=False;self.log.setText('RESET COMPLETE  • Simulation restarted')
 def load_video(self):
  p,_=QFileDialog.getOpenFileName(self,'Load BP-2 Video','','MP4 Video (*.mp4);;All Files (*)')
  if p:self.source=VideoFileSource(p);self.is_video_source=True;self.state='TRACKING';self.recent_errors.clear();self.log.setText('BP-2 PIPELINE  • Video source loaded')
 def export_report(self):
  r,c,j=self.logger.generate_summary();b=r['performance_benchmarks'];QMessageBox.information(self,'Benchmark Report Exported',f'CSV: {c}\nJSON: {j}\n\nRMSE: {b["rmse_tracking_error_px"]} px\nLock Rate: {b["lock_retention_rate_percent"]}%')
 def pix(self,f,w):
  rgb=cv2.cvtColor(f,cv2.COLOR_BGR2RGB);h,x=rgb.shape[:2];q=QImage(rgb.data,x,h,3*x,QImage.Format_RGB888).copy();return QPixmap.fromImage(q).scaled(w.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation)
 def world_view(self):
  d=cv2.cvtColor(self.source._build_full_scene(),cv2.COLOR_GRAY2BGR);tx,ty=int(self.source.target_x),int(self.source.target_y);cv2.circle(d,(tx,ty),16,(0,255,255),-1);z=self.source.fov_deg/4;cw=int(640*z);ch=int(480*z);x=int(self.source.cam_x-cw/2);y=int(self.source.cam_y-ch/2);cv2.rectangle(d,(x,y),(x+cw,y+ch),(0,0,255),6);d=cv2.resize(d,(480,480));return cv2.copyMakeBorder(d,0,0,80,80,cv2.BORDER_CONSTANT,value=(15,23,42))
 def update_loop(self):
  if not self.running:return
  t=time.time();self.frame_id+=1;frame=self.source.get_frame();
  if frame is None and self.is_video_source:self.export_report();self.running=False;return
  px,py=self.kf.predict();cx,cy,_=self.detector.detect(frame);mx=(self.pan_speed_deg/FPS)*(VIEWPORT_WIDTH/self.fov_deg);err=ke=None
  if cx is not None:
   self.frames_lost=0;kx,ky=self.kf.update(cx,cy);ex,ey=kx-SETPOINT_X,ky-SETPOINT_Y;ke=err=math.hypot(ex,ey)
   if self.state in ('SEARCHING','ACQUIRING'):
    self.search_frames+=1
    if ke<=10:
     if self.acquisition_time is None:self.acquisition_time=self.search_frames/FPS;self.logger.record_acquisition(self.acquisition_time)
     self.state='TRACKING';self.recent_errors.clear()
    else:self.state='ACQUIRING'
   if self.state=='TRACKING':self.recent_errors.append(ke);self.recent_errors=self.recent_errors[-300:]
   dx,dy=self.pid.compute(ex,ey,mx)
   if not self.is_video_source:self.source.pan_tilt(dx,dy)
  else:
   if self.state=='TRACKING':
    self.frames_lost+=1
    if self.frames_lost<=15:
     dx,dy=self.pid.compute(px-SETPOINT_X,py-SETPOINT_Y,mx)
     if not self.is_video_source:self.source.pan_tilt(dx,dy)
    else:self.state='SEARCHING';self.pid.reset();self.kf.reset();self.scanner.center_x=self.source.cam_x;self.scanner.center_y=self.source.cam_y;self.scanner.r=0;self.scanner.theta=0
   if self.state=='SEARCHING' and not self.is_video_source:self.search_frames+=1;nx,ny=self.scanner.next_position();self.source.set_position(nx,ny)
  pan=(self.source.cam_x-self.source.scene_width/2)/PIXELS_PER_DEGREE if not self.is_video_source else 0;tilt=(self.source.cam_y-self.source.scene_height/2)/PIXELS_PER_DEGREE if not self.is_video_source else 0;lat=(time.time()-t)*1000;dt=time.time()-self.last_time;self.fps=.9*self.fps+.1*(1/dt) if dt>0 else self.fps;self.last_time=time.time();target=(self.source.target_x,self.source.target_y) if not self.is_video_source else (0,0);cam=(self.source.cam_x,self.source.cam_y) if not self.is_video_source else (0,0);self.logger.log_frame(self.frame_id,self.state,self.motion.currentText(),target,cam,pan,tilt,err,ke,self.fps,lat,self.source.noise_mode,self.source.atmosphere_mode)
  d=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR);cv2.drawMarker(d,(SETPOINT_X,SETPOINT_Y),(100,100,100),cv2.MARKER_CROSS,20,1);cv2.circle(d,(SETPOINT_X,SETPOINT_Y),10,(60,60,60),1)
  if cx is not None:
   cv2.circle(d,(int(cx),int(cy)),4,(255,255,0),1);col=(0,255,0) if err is not None and err<=10 else (0,0,255);cv2.circle(d,(int(px),int(py)),7,col,2);cv2.line(d,(SETPOINT_X,SETPOINT_Y),(int(px),int(py)),(0,165,255),1)
  self.cam.setPixmap(self.pix(d,self.cam));
  if not self.is_video_source:self.world.setPixmap(self.pix(self.world_view(),self.world))
  a=np.array(self.recent_errors) if self.recent_errors else np.array([0.]);rmse=float(np.sqrt(np.mean(a*a)));lock=float(np.mean(a<=10)*100);col={'TRACKING':'#4ade80','ACQUIRING':'#63d9ff','SEARCHING':'#facc15'}.get(self.state,'#f87171');self.m_state.set(self.state,col);self.m_err.set('SEARCHING...' if err is None else f'{err:.2f} px  '+('✓' if err<=10 else '✕'));self.m_acq.set('—' if self.acquisition_time is None else f'{self.acquisition_time:.2f}s');self.m_rmse.set(f'{rmse:.2f} px');self.m_lock.set(f'{lock:.1f}%');self.m_perf.set(f'{self.fps:.1f} / {lat:.1f} ms');self.status.setText('● '+self.state)
 def closeEvent(self,e):
  try:self.logger.generate_summary()
  except Exception:pass
  e.accept()
def main():
 app=QApplication(sys.argv);app.setStyleSheet(STYLE);w=FSOCWindow();w.show();sys.exit(app.exec())
if __name__=='__main__':main()
