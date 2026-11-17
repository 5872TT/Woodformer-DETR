import sys
from pathlib import Path

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QColor, QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QGridLayout, QGroupBox, QSlider, QTabWidget, QTableWidget,
    QTableWidgetItem, QTextEdit, QFileDialog, QListWidget, QCheckBox,
    QComboBox, QSplitter, QFrame, QStatusBar, QAction, QMenu, QScrollArea,
    QSizePolicy,
)


class ImageView(QLabel):
    """Dark image viewport; the model output can later be painted on this widget."""
    def __init__(self):
        super().__init__("暂无图片")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(180, 135)
        self.setStyleSheet("background:#111a22;color:#8795a3;border:1px solid #65717c")
        self._pixmap = None

    def set_image(self, path):
        pix = QPixmap(path)
        if pix.isNull():
            return False
        self._pixmap = pix
        self._refresh()
        return True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self):
        if self._pixmap:
            self.setPixmap(self._pixmap.scaled(self.size() - QSize(10, 10), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.setText("")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("木材表面缺陷检测系统")
        self.resize(1000, 760)
        self.setMinimumSize(760, 560)
        self.image_path = None
        self.weights = []
        self._build_menus()
        self._build_ui()
        self._apply_style()

    def _build_menus(self):
        bar = self.menuBar()
        for title in ("文件(&F)", "编辑(&E)", "视图(&V)", "工具(&T)", "帮助(&H)"):
            bar.addMenu(title)
        toolbar = self.addToolBar("快捷操作")
        toolbar.setMovable(False)
        for label, slot in (("打开", self.open_image), ("保存", self.save_result),
                            ("开始检测", self.detect), ("导出检测", self.save_result),
                            ("切换主题", self.toggle_theme), ("设置窗口", self.show_settings)):
            action = QAction(label, self)
            action.triggered.connect(slot)
            toolbar.addAction(action)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(7, 6, 7, 5)
        outer.setSpacing(5)

        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)
        self.left = self._left_panel()
        self.center = self._center_panel()
        self.right = self._right_panel()
        split.addWidget(self.left)
        split.addWidget(self.center)
        split.addWidget(self.right)
        split.setSizes([215, 555, 200])
        outer.addWidget(split, 1)

        log_box = QGroupBox("日志")
        log_l = QVBoxLayout(log_box)
        log_l.setContentsMargins(4, 4, 4, 4)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(105)
        log_l.addWidget(self.log)
        outer.addWidget(log_box)
        self.statusBar().showMessage("就绪")
        self._log("系统已启动，等待加载模型和图片。")

    def _left_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(170)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        files = QGroupBox("文件管理")
        fl = QVBoxLayout(files)
        for text, fn in (("打开图片", self.open_image), ("打开视频", self.open_video), ("打开文件夹", self.open_folder)):
            btn = QPushButton(text); btn.clicked.connect(fn); fl.addWidget(btn)
        layout.addWidget(files)

        model = QGroupBox("模型选择")
        ml = QVBoxLayout(model)
        row = QHBoxLayout()
        for text, fn in (("选择模型权重", self.open_weights), ("选择模型文件夹", self.open_weight_folder)):
            b = QPushButton(text); b.clicked.connect(fn); row.addWidget(b)
        ml.addLayout(row)
        self.model_label = QLabel("当前模型：未加载")
        ml.addWidget(self.model_label)
        self.weight_list = QListWidget(); self.weight_list.setMaximumHeight(65)
        ml.addWidget(self.weight_list)
        layout.addWidget(model)

        conf = QGroupBox("检测阈值")
        cl = QGridLayout(conf)
        cl.addWidget(QLabel("置信度阈值"), 0, 0, 1, 2)
        self.conf_slider = self._slider(50)
        self.conf_value = QLabel("0.50")
        self.conf_slider.valueChanged.connect(lambda v: self.conf_value.setText(f"{v/100:.2f}"))
        cl.addWidget(self.conf_slider, 1, 0); cl.addWidget(self.conf_value, 1, 1)
        cl.addWidget(QLabel("IOU阈值"), 2, 0, 1, 2)
        self.iou_slider = self._slider(45)
        self.iou_value = QLabel("0.45")
        self.iou_slider.valueChanged.connect(lambda v: self.iou_value.setText(f"{v/100:.2f}"))
        cl.addWidget(self.iou_slider, 3, 0); cl.addWidget(self.iou_value, 3, 1)
        layout.addWidget(conf)
        layout.addStretch(1)
        return panel

    def _center_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(5)
        self.tabs = QTabWidget()
        viewpage = QWidget(); vl = QVBoxLayout(viewpage); vl.setContentsMargins(4, 4, 4, 4)
        self.image_view = ImageView(); vl.addWidget(self.image_view)
        self.tabs.addTab(viewpage, "图片检测")
        self.tabs.addTab(QWidget(), "批量处理")
        layout.addWidget(self.tabs, 1)
        self.detect_button = QPushButton("开始检测")
        self.detect_button.clicked.connect(self.detect)
        layout.addWidget(self.detect_button)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["类别", "置信度", "位置", "大小", "ID"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(90)
        layout.addWidget(self.table)
        return panel

    def _right_panel(self):
        panel = QWidget(); panel.setMinimumWidth(160)
        layout = QVBoxLayout(panel); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(6)
        info = QGroupBox("检测状态")
        il = QVBoxLayout(info)
        self.count_label = QLabel("总检测数量：0")
        self.time_label = QLabel("检测耗时：0.0 ms")
        il.addWidget(self.count_label); il.addWidget(self.time_label)
        layout.addWidget(info)
        classes = QGroupBox("类别勾选")
        ql = QVBoxLayout(classes)
        self.class_checks = []
        for name in ("dry_knot", "sound_knot", "edge_knot"):
            cb = QCheckBox(name); cb.setChecked(True); self.class_checks.append(cb); ql.addWidget(cb)
        layout.addWidget(classes)
        display = QGroupBox("显示设置")
        dl = QVBoxLayout(display)
        for name in ("显示检测框", "显示标签", "显示置信度"):
            cb = QCheckBox(name); cb.setChecked(True); dl.addWidget(cb)
        dl.addWidget(QLabel("边界框颜色："))
        self.color_combo = QComboBox(); self.color_combo.addItems(["绿色", "红色", "蓝色", "黄色"])
        dl.addWidget(self.color_combo)
        layout.addWidget(display)
        layout.addStretch(1)
        return panel

    @staticmethod
    def _slider(value):
        s = QSlider(Qt.Horizontal); s.setRange(0, 100); s.setValue(value)
        return s

    def _log(self, text):
        self.log.append(text)

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp *.webp)")
        if path:
            self.load_image(path)

    def load_image(self, path):
        if self.image_view.set_image(path):
            self.image_path = path
            self._log(f"已打开图片：{Path(path).name}")
            self.statusBar().showMessage(f"当前图片：{path}")

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)")
        if path: self._log(f"已选择视频：{Path(path).name}（视频检测模块待接入）")

    def open_folder(self):
        path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if path:
            files = sorted(p for p in Path(path).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"})
            if files: self.load_image(str(files[0])); self._log(f"文件夹中找到 {len(files)} 张图片")

    def open_weights(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "选择模型权重", "", "模型权重 (*.pt *.onnx *.engine);;所有文件 (*)")
        if paths:
            self.weights = paths; self.weight_list.clear(); self.weight_list.addItems([Path(p).name for p in paths])
            self.model_label.setText(f"当前模型：{Path(paths[0]).name}")
            self._log(f"已选择模型权重：{paths[0]}（推理引擎待接入）")

    def open_weight_folder(self):
        path = QFileDialog.getExistingDirectory(self, "选择模型文件夹")
        if path:
            files = sorted(Path(path).glob("*.pt"))
            if files:
                self.weights = [str(p) for p in files]; self.weight_list.clear(); self.weight_list.addItems([p.name for p in files])
                self.model_label.setText(f"当前模型：{files[0].name}")

    def detect(self):
        if not self.image_path:
            self._log("请先打开一张图片。")
        elif not self.weights:
            self._log("图片已就绪；请上传并选择训练好的模型权重后接入推理。")
        else:
            self._log("检测界面已就绪，模型推理接口待接入。")

    def save_result(self):
        if not self.image_path:
            self._log("当前没有可保存的检测结果。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存结果图片", "detection_result.png", "PNG图片 (*.png);;JPEG图片 (*.jpg)")
        if path:
            self.image_view._pixmap.save(path)
            self._log(f"已保存图片：{path}")

    def toggle_theme(self):
        self._log("当前界面使用深色主题。")

    def show_settings(self):
        self._log("当前检测参数可通过置信度和 IOU 滑块调整。")

    def _apply_style(self):
        self.setStyleSheet("""
            QWidget { background:#273441; color:#d7e0e8; font: 12px 'Microsoft YaHei UI','Microsoft YaHei',sans-serif; }
            QMainWindow { background:#273441; }
            QMenuBar, QToolBar { background:#344454; color:#e3e9ef; border:0; spacing:5px; }
            QMenuBar::item { padding:4px 6px; } QToolBar { padding:3px; }
            QToolButton { color:#e0e6ec; padding:3px 6px; }
            QGroupBox { border:1px solid #536270; border-radius:3px; margin-top:8px; padding-top:5px; font-weight:bold; }
            QGroupBox::title { subcontrol-origin:margin; left:6px; padding:0 3px; }
            QPushButton { background:#47596a; border:1px solid #5b6c7a; border-radius:3px; padding:5px 7px; }
            QPushButton:hover { background:#566a7c; } QPushButton:pressed { background:#344554; }
            QTabWidget::pane { border:1px solid #5b6874; } QTabBar::tab { background:#344454; padding:4px 9px; border:1px solid #536270; }
            QTabBar::tab:selected { background:#455767; }
            QListWidget, QTextEdit, QTableWidget { background:#1c2833; border:1px solid #465563; alternate-background-color:#253542; }
            QHeaderView::section { background:#394b5b; color:#e0e7ed; border:0; padding:4px; }
            QSlider::groove:horizontal { height:3px; background:#536575; } QSlider::handle:horizontal { background:#a9bac7; width:9px; margin:-4px 0; border-radius:4px; }
            QCheckBox { spacing:5px; } QStatusBar { background:#344454; color:#d2dbe2; }
        """)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 9))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
