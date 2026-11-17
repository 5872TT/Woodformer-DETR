"""RT-DETR inference application. UI is inherited unchanged from main.py."""
import sys
import csv
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if sys.version_info[:2] == (3, 11):
    sys.path.insert(0, str(ROOT / '.runtime'))
from main import MainWindow, ImageView
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QSize
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont, QIcon
from PyQt5.QtWidgets import (QApplication, QFileDialog, QTableWidgetItem, QCheckBox, QToolBar,
    QSplitter, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QListWidget, QListWidgetItem, QAbstractItemView)

CLASSES = ['dry_knot', 'sound_knot', 'edge_knot']
PROJECT = ROOT / 'RT-DETR-main/rtdetr_pytorch'
BEST = ROOT / 'weights/best.pt'

def initialize_runtime():
    # Initialize native runtime imports on the main thread before starting Qt's
    # event loop. Checkpoint I/O and inference remain on the worker thread.
    import torch
    import numpy
    from PIL import Image
    sys.path.insert(0, str(PROJECT))
    from src.core import YAMLConfig

class Engine:
    def __init__(self):
        self.model = None
        self.path = None

    def load(self, path):
        import torch
        sys.path.insert(0, str(PROJECT))
        from src.core import YAMLConfig
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        state = checkpoint.get('ema', {}).get('module') if isinstance(checkpoint, dict) else None
        source = 'EMA'
        if state is None:
            state = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
            source = 'model'
        kind = 'fewd' if any(k.startswith('backbone.stem.') for k in state) else 'baseline'
        cfg = YAMLConfig(str(PROJECT / f'configs/inference/{kind}.yml'))
        # Strict loading rejects unsupported architectures/classes instead of silently using random parameters.
        model = cfg.model
        model.load_state_dict(state, strict=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.eval().to(device).deploy()
        post = cfg.postprocessor.eval().to(device).deploy()
        self.model, self.post, self.device = model, post, device
        self.size = cfg.yaml_cfg['RTDETRTransformer']['eval_spatial_size']
        self.path = str(path)
        return f'{kind} / {source} / {device}'

    def predict(self, path):
        import torch
        import numpy as np
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert('RGB'); w,h = im.size
            a = np.array(im.resize((self.size[1],self.size[0]),Image.Resampling.BILINEAR),dtype=np.float32)/255
        x = torch.from_numpy(a).permute(2,0,1).unsqueeze(0).to(self.device)
        if self.device.type == 'cuda': torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            labels,boxes,scores = self.post(self.model(x),torch.tensor([[w,h]],device=self.device))
        if self.device.type == 'cuda': torch.cuda.synchronize()
        elapsed = (time.perf_counter()-start)*1000
        rows=[]
        for label,box,score in zip(labels[0].cpu().tolist(),boxes[0].cpu().tolist(),scores[0].cpu().tolist()):
            x1,y1,x2,y2 = box
            rows.append({'label':label,'score':score,'box':[max(0,min(w,x1)),max(0,min(h,y1)),max(0,min(w,x2)),max(0,min(h,y2))]})
        return rows,elapsed

class Job(QThread):
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    def __init__(self, fn):
        super().__init__(); self.fn=fn
    def run(self):
        try: self.result.emit(self.fn())
        except Exception as exc: self.error.emit(f'{type(exc).__name__}: {exc}')

class DetectionWindow(MainWindow):
    def __init__(self):
        self.results={}
        self.completed_detections=0
        self.inference_active=False
        self.engine=Engine(); self.job=None; self.pending=False; self.raw=[]; self.visible=[]; self.folder=[]; self.queue=[]; self.batch_active=False
        super().__init__()
        self._log('运行环境：'+sys.executable)
        # QLabel.setText after setPixmap clears the pixmap in the original viewer.
        self.image_view._refresh = self.refresh_preview
        self.display_checks={c.text():c for c in self.right.findChildren(QCheckBox)}
        for c in self.display_checks.values(): c.toggled.connect(self.render_result)
        self.conf_slider.valueChanged.connect(self.render_result)
        self.color_combo.currentIndexChanged.connect(self.render_result)
        self.iou_slider.setToolTip('RT-DETR 为端到端集合预测，无需 NMS，IOU 滑块不参与推理。')
        self.iou_slider.sliderReleased.connect(lambda: self._log('RT-DETR 不使用 NMS，IOU 滑块不参与当前模型推理。'))
        self.weight_list.currentRowChanged.connect(self.select_weight)
        for toolbar in self.findChildren(QToolBar):
            for action in toolbar.actions():
                if action.text() == '导出检测':
                    action.triggered.disconnect(); action.triggered.connect(self.export_csv)
        if BEST.exists():
            self.weights=[str(BEST)]; self.weight_list.addItem(BEST.name)
            self._log('已找到推荐权重，选择列表中的权重即可加载。')

    def _build_ui(self):
        super()._build_ui()
        # Keep the result table compact so the image preview gets spare height.
        self.table.setFixedHeight(120)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        outer=self.centralWidget().layout()
        content=outer.takeAt(0).widget()
        log_group=outer.takeAt(0).widget()
        self.vertical_split=QSplitter(Qt.Vertical)
        self.vertical_split.setChildrenCollapsible(False)
        self.vertical_split.setHandleWidth(7)
        self.vertical_split.addWidget(content)
        self.vertical_split.addWidget(log_group)
        self.log.setMaximumHeight(16777215)
        self.log.setMinimumHeight(90)
        self.log.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.vertical_split.setSizes([460,210])
        self.vertical_split.setStretchFactor(0,3)
        self.vertical_split.setStretchFactor(1,1)
        self.vertical_split.handle(1).setToolTip('上下拖动，调整日志区域高度')
        self.vertical_split.setStyleSheet('QSplitter::handle:vertical { background:#536270; }')
        outer.addWidget(self.vertical_split,1)
        page=self.tabs.widget(1)
        layout=QHBoxLayout(page);layout.setContentsMargins(4,4,4,4);layout.setSpacing(5)
        strip=QWidget();strip.setFixedWidth(120)
        sl=QVBoxLayout(strip);sl.setContentsMargins(0,0,0,0);sl.setSpacing(3)
        choose=QPushButton('选择文件夹');choose.clicked.connect(self.open_folder);sl.addWidget(choose)
        self.batch_list=QListWidget()
        self.batch_list.setIconSize(QSize(84,70))
        self.batch_list.setViewMode(QListWidget.IconMode)
        self.batch_list.setFlow(QListWidget.TopToBottom)
        self.batch_list.setWrapping(False)
        self.batch_list.setMovement(QListWidget.Static)
        self.batch_list.setGridSize(QSize(100,100))
        self.batch_list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.batch_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.batch_list.currentRowChanged.connect(self.browse_batch)
        sl.addWidget(self.batch_list,1)
        layout.addWidget(strip)
        self.batch_view=ImageView()
        self.batch_view._refresh=self.refresh_batch_preview
        layout.addWidget(self.batch_view,1)

    def _log(self,text):
        # Preserve the reader's position when inspecting earlier messages.
        bar=self.log.verticalScrollBar()
        follow=bar.value()>=bar.maximum()-2
        position=bar.value()
        self.log.append(text)
        bar.setValue(bar.maximum() if follow else position)

    def refresh_batch_preview(self):
        pix=self.batch_view._pixmap
        if pix is not None:
            self.batch_view.setPixmap(pix.scaled(self.batch_view.size()-QSize(10,10),Qt.KeepAspectRatio,Qt.SmoothTransformation))

    def browse_batch(self,row):
        if 0<=row<len(self.folder) and not self.busy():
            self.load_image(self.folder[row])

    def refresh_preview(self):
        pix=self.image_view._pixmap
        if pix is not None:
            self.image_view.setPixmap(pix.scaled(self.image_view.size()-QSize(10,10),Qt.KeepAspectRatio,Qt.SmoothTransformation))

    def busy(self):
        return self.pending

    def run_job(self, fn, callback):
        if self.busy():
            self._log('任务进行中，请稍候。'); return
        self.job=Job(fn)
        self.pending=True
        self.job.result.connect(callback)
        self.job.error.connect(self.failed)
        self.job.finished.connect(self.finished)
        self.detect_button.setEnabled(False)
        self.batch_list.setEnabled(False)
        self.job.start()

    def failed(self, message):
        self.inference_active=False
        self.render_result()
        self.queue=[]; self.batch_active=False; self._log('操作失败：'+message)
        self.statusBar().showMessage('操作失败，请查看日志')

    def finished(self):
        self.pending=False
        self.detect_button.setEnabled(True)
        self.batch_list.setEnabled(True)
        if self.queue:
            path=self.queue.pop(0); self.detect_one(path)
        elif self.batch_active:
            self.batch_active=False
            self.render_result()
            self._log('批量检测完成，结果保存在 detection_results 文件夹。')

    def select_weight(self, row):
        if row < 0 or row >= len(self.weights): return
        if self.busy(): self._log('请等待当前任务结束后再选择权重。'); return
        path=self.weights[row]
        self._log('正在加载：'+path)
        def done(info):
            self.model_label.setText('当前模型：'+Path(path).name)
            self.results.clear(); self.raw=[]
            for i,image_path in enumerate(self.folder):
                self.batch_list.item(i).setText(Path(image_path).name)
            self.render_result(); self._log('模型加载完成：'+info)
        # Construct the CUDA model on the main thread. Inference continues in
        # workers; this avoids native initialization failures during weight load.
        self.pending=True
        self.detect_button.setEnabled(False)
        self.batch_list.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            done(self.engine.load(path))
        except Exception as exc:
            self.failed(f'{type(exc).__name__}: {exc}')
        finally:
            QApplication.restoreOverrideCursor()
            self.pending=False
            self.detect_button.setEnabled(True)
            self.batch_list.setEnabled(True)

    def set_weights(self, paths):
        if self.busy(): self._log('请等待当前任务完成。'); return
        self.weight_list.blockSignals(True)
        self.weights=list(map(str,paths)); self.weight_list.clear()
        self.weight_list.addItems([Path(p).name for p in paths])
        self.weight_list.setCurrentRow(0); self.weight_list.blockSignals(False)
        self.select_weight(0)

    def open_weights(self):
        paths,_=QFileDialog.getOpenFileNames(self,'选择训练后的权重',str(BEST.parent),'PyTorch 权重 (*.pth *.pt)')
        if paths: self.set_weights(paths)

    def open_weight_folder(self):
        path=QFileDialog.getExistingDirectory(self,'选择权重文件夹',str(BEST.parent))
        if path:
            files=sorted(p for p in Path(path).iterdir() if p.suffix.lower() in ('.pth','.pt'))
            if files: self.set_weights(files)
            else: self._log('该文件夹没有 .pth/.pt 权重。')

    def load_image(self,path):
        if self.busy(): self._log('请等待当前任务完成再切换图片。'); return
        if QPixmap(path).isNull():
            self._log('无法读取图片：'+str(path)); self.queue=[]; return
        super().load_image(path)
        self.raw,elapsed=self.results.get(str(path),([],0.))
        self.render_result(); self.time_label.setText(f'检测耗时：{elapsed:.1f} ms')
        if str(path) in self.folder:
            self.batch_list.blockSignals(True)
            self.batch_list.setCurrentRow(self.folder.index(str(path)))
            self.batch_list.scrollToItem(self.batch_list.currentItem())
            self.batch_list.blockSignals(False)

    def open_folder(self):
        if self.busy(): self._log('请等待当前任务完成。'); return
        path=QFileDialog.getExistingDirectory(self,'选择图片文件夹')
        if path: self.set_folder(path)

    def set_folder(self,path):
        if self.busy(): return
        self.folder=[str(p) for p in sorted(Path(path).iterdir()) if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.webp')]
        self.batch_list.blockSignals(True);self.batch_list.clear()
        for file in self.folder:
            pix=QPixmap(file)
            item=QListWidgetItem(QIcon(pix.scaled(84,70,Qt.KeepAspectRatio,Qt.SmoothTransformation)),Path(file).name)
            item.setToolTip(file);self.batch_list.addItem(item)
        self.batch_list.blockSignals(False)
        self.tabs.setCurrentIndex(1)
        if self.folder:
            self.load_image(self.folder[0]); self._log(f'找到 {len(self.folder)} 张图，点击开始检测。检测后可点击缩略图回看。')
        else:
            self.image_path=None;self.raw=[];self.visible=[];self.table.setRowCount(0)
            self.batch_view._pixmap=None;self.batch_view.setText('文件夹中没有图片')
            self._log('文件夹中没有支持的图片。')

    def detect(self):
        if self.busy(): return
        if self.engine.model is None:
            self._log('请先选择权重并等待模型加载完成。'); return
        if self.tabs.currentIndex()==1:
            if not self.folder: self._log('请先选择图片文件夹。'); return
            self.batch_active=True
            self.queue=self.folder[1:].copy()
            self.detect_one(self.folder[0])
            return
        self.detect_one()

    def detect_one(self, path=None):
        path=path or self.image_path
        if not path: self._log('请先打开图片。'); return
        self.inference_active=True
        self.table.setRowCount(0)
        self.statusBar().showMessage(f'正在检测：{Path(path).name}')
        def done(result):
            self.inference_active=False
            self.completed_detections+=1
            self.image_path=path
            self.results[path]=result
            if path in self.folder:
                self.batch_list.blockSignals(True)
                self.batch_list.setCurrentRow(self.folder.index(path))
                self.batch_list.scrollToItem(self.batch_list.currentItem())
                self.batch_list.blockSignals(False)
            self.raw,elapsed=result; self.render_result()
            self.time_label.setText(f'检测耗时：{elapsed:.1f} ms')
            self._log(f'{Path(path).name}：{len(self.visible)} 个目标，{elapsed:.1f} ms')
            self.statusBar().showMessage(f'检测完成：{Path(path).name}')
            if self.batch_active:
                index=self.folder.index(path)
                self.batch_list.item(index).setText(Path(path).name+'\n已检测')
                self.statusBar().showMessage(f'批量检测：{index+1}/{len(self.folder)}  {Path(path).name}')
                out=ROOT/'detection_results'; out.mkdir(exist_ok=True)
                destination=out/(Path(path).name+'_detected.png')
                if not self.image_view._pixmap.save(str(destination)):
                    self._log('图片保存失败：'+str(destination))
        self.run_job(lambda:self.engine.predict(path),done)

    def render_result(self,*args):
        self.table.setRowCount(0)
        if self.inference_active:
            return
        if not self.image_path: return
        pix=QPixmap(self.image_path)
        if pix.isNull(): return
        enabled={i for i,c in enumerate(self.class_checks) if c.isChecked()}
        self.visible=[r for r in self.raw if r['label'] in enabled and r['score']>=self.conf_slider.value()/100]
        painter=QPainter(pix); painter.setPen(QColor(['#28e65a','#ff4040','#4488ff','#ffff00'][self.color_combo.currentIndex()]))
        painter.setFont(QFont('Arial',10))
        def checked(name): return self.display_checks[name].isChecked()
        show_details=not self.batch_active
        self.table.setRowCount(len(self.visible) if show_details else 0)
        for i,r in enumerate(self.visible):
            x1,y1,x2,y2=r['box']; name=CLASSES[r['label']]
            if checked('显示检测框'): painter.drawRect(int(x1),int(y1),int(x2-x1),int(y2-y1))
            label=[]
            if checked('显示标签'): label.append(name)
            if checked('显示置信度'): label.append(f"{r['score']:.2f}")
            if label: painter.drawText(int(x1),max(14,int(y1)-3),' '.join(label))
            if show_details:
                values=[name,f"{r['score']:.3f}",f'({x1:.0f},{y1:.0f})',
                        f'{x2-x1:.0f}×{y2-y1:.0f}',str(i+1)]
                for column,value in enumerate(values):
                    item=QTableWidgetItem(value)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.table.setItem(i,column,item)
        painter.end()
        self.image_view._pixmap=pix; self.image_view._refresh()
        self.batch_view._pixmap=pix;self.batch_view._refresh()
        self.count_label.setText(f'总检测数量：{self.completed_detections}')

    def export_csv(self):
        if self.inference_active:
            self._log('正在检测，请等待真实结果生成后导出。'); return
        if not self.image_path: return
        path,_=QFileDialog.getSaveFileName(self,'导出检测结果','detections.csv','CSV (*.csv)')
        if path:
            with open(path,'w',newline='',encoding='utf-8-sig') as f:
                writer=csv.writer(f);writer.writerow(['image','class','confidence','xmin','ymin','xmax','ymax'])
                for r in self.visible:writer.writerow([self.image_path,CLASSES[r['label']],r['score'],*r['box']])
            self._log('已导出：'+path)

    def open_video(self):
        self._log('此版本支持图片及文件夹检测，视频检测暂未接入。')

    def save_result(self):
        if self.image_view._pixmap is None:
            self._log('请先打开图片并完成检测。'); return
        path,_=QFileDialog.getSaveFileName(self,'保存结果图片','detection_result.png','PNG (*.png);;JPEG (*.jpg)')
        if path:
            if self.image_path and Path(path).resolve()==Path(self.image_path).resolve():
                self._log('请选择新的文件名以保留原图。'); return
            ok=self.image_view._pixmap.save(path)
            self._log(('已保存：' if ok else '保存失败：')+path)

    def closeEvent(self,event):
        if self.busy():
            self._log('模型任务正在执行，请等待结束后关闭窗口。');event.ignore()
        else: event.accept()

if __name__=='__main__':
    initialize_runtime()
    app=QApplication(sys.argv);app.setStyle('Fusion');app.setFont(QFont('Microsoft YaHei UI',9))
    window=DetectionWindow();window.show();sys.exit(app.exec_())
