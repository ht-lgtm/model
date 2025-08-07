import customtkinter as ctk
from tkinter import filedialog, messagebox, Tk
import os
import shutil
from PIL import Image, ImageTk
import cv2
import numpy as np
import albumentations as A
import random

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("YOLO 데이터 편집 및 증강 도구")
        self.geometry("1300x800")

        # --- 메인 프레임 설정 ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- 왼쪽 컨트롤 프레임 ---
        self.left_frame = ctk.CTkFrame(self, width=320, corner_radius=0)
        self.left_frame.grid(row=0, column=0, sticky="nsw")
        self.left_frame.grid_rowconfigure(21, weight=1) # 로그 박스가 남은 공간을 채우도록 설정

        self.left_frame_title = ctk.CTkLabel(self.left_frame, text="설정", font=ctk.CTkFont(size=20, weight="bold"))
        self.left_frame_title.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 10))

        # --- 폴더 선택 섹션 ---
        self.image_dir_button = ctk.CTkButton(self.left_frame, text="이미지 폴더 선택", command=self.select_image_dir)
        self.image_dir_button.grid(row=1, column=0, columnspan=2, padx=20, pady=5, sticky="ew")
        self.image_dir_label = ctk.CTkLabel(self.left_frame, text="선택되지 않음", wraplength=280)
        self.image_dir_label.grid(row=2, column=0, columnspan=2, padx=20, pady=(0, 5))

        self.label_dir_button = ctk.CTkButton(self.left_frame, text="레이블 폴더 선택", command=self.select_label_dir)
        self.label_dir_button.grid(row=3, column=0, columnspan=2, padx=20, pady=5, sticky="ew")
        self.label_dir_label = ctk.CTkLabel(self.left_frame, text="선택되지 않음", wraplength=280)
        self.label_dir_label.grid(row=4, column=0, columnspan=2, padx=20, pady=(0, 5))

        self.output_dir_button = ctk.CTkButton(self.left_frame, text="출력 폴더 선택", command=self.select_output_dir)
        self.output_dir_button.grid(row=5, column=0, columnspan=2, padx=20, pady=5, sticky="ew")
        self.output_dir_label = ctk.CTkLabel(self.left_frame, text="선택되지 않음", wraplength=280)
        self.output_dir_label.grid(row=6, column=0, columnspan=2, padx=20, pady=(0, 10))

        # --- 클래스 선택 섹션 ---
        self.class_label = ctk.CTkLabel(self.left_frame, text="클래스 선택", font=ctk.CTkFont(size=16, weight="bold"))
        self.class_label.grid(row=7, column=0, columnspan=2, padx=20, pady=(15, 5))
        self.class_var = ctk.StringVar(value="클래스를 불러오세요")
        self.class_menu = ctk.CTkOptionMenu(self.left_frame, variable=self.class_var, values=["클래스를 불러오세요"], command=self.on_class_change)
        self.class_menu.grid(row=8, column=0, columnspan=2, padx=20, pady=5, sticky="ew")

        # --- 증강 옵션 섹션 ---
        self.aug_options_label = ctk.CTkLabel(self.left_frame, text="증강 옵션", font=ctk.CTkFont(size=16, weight="bold"))
        self.aug_options_label.grid(row=9, column=0, columnspan=2, padx=20, pady=(15, 5))

        self.flip_h_var = ctk.BooleanVar()
        self.flip_h_check = ctk.CTkCheckBox(self.left_frame, text="좌우 대칭", variable=self.flip_h_var)
        self.flip_h_check.grid(row=10, column=0, columnspan=2, padx=20, pady=5, sticky="w")

        self.flip_v_var = ctk.BooleanVar()
        self.flip_v_check = ctk.CTkCheckBox(self.left_frame, text="상하 대칭", variable=self.flip_v_var)
        self.flip_v_check.grid(row=11, column=0, columnspan=2, padx=20, pady=5, sticky="w")

        self.rot_var = ctk.BooleanVar()
        self.rot_check = ctk.CTkCheckBox(self.left_frame, text="회전", variable=self.rot_var)
        self.rot_check.grid(row=12, column=0, padx=20, pady=5, sticky="w")
        self.rot_entry = ctk.CTkEntry(self.left_frame, placeholder_text="각도 (예: 15)")
        self.rot_entry.grid(row=12, column=1, padx=20, pady=5, sticky="we")

        self.scale_var = ctk.BooleanVar()
        self.scale_check = ctk.CTkCheckBox(self.left_frame, text="크기 조절", variable=self.scale_var)
        self.scale_check.grid(row=13, column=0, padx=20, pady=5, sticky="w")
        self.scale_entry = ctk.CTkEntry(self.left_frame, placeholder_text="비율 (예: 0.1)")
        self.scale_entry.grid(row=13, column=1, padx=20, pady=5, sticky="we")

        # --- 실행 버튼 ---
        self.run_button = ctk.CTkButton(self.left_frame, text="증강 시작", command=self.augment_and_save, height=40, font=ctk.CTkFont(size=16, weight="bold"))
        self.run_button.grid(row=15, column=0, columnspan=2, padx=20, pady=(20, 10), sticky="ew")

        # --- 데이터셋 분할 섹션 ---
        self.split_dataset_label = ctk.CTkLabel(self.left_frame, text="데이터셋 분할", font=ctk.CTkFont(size=16, weight="bold"))
        self.split_dataset_label.grid(row=17, column=0, columnspan=2, padx=20, pady=(20, 5))

        self.source_dir_button = ctk.CTkButton(self.left_frame, text="분할할 폴더 선택", command=self.select_source_dir)
        self.source_dir_button.grid(row=18, column=0, columnspan=2, padx=20, pady=5, sticky="ew")
        self.source_dir_label = ctk.CTkLabel(self.left_frame, text="선택되지 않음", wraplength=280)
        self.source_dir_label.grid(row=19, column=0, columnspan=2, padx=20, pady=(0, 10))

        self.split_run_button = ctk.CTkButton(self.left_frame, text="분할 시작", command=self.split_dataset, height=40, font=ctk.CTkFont(size=16, weight="bold"))
        self.split_run_button.grid(row=20, column=0, columnspan=2, padx=20, pady=(10, 10), sticky="ew")

        # --- 로그 박스 ---
        self.log_box = ctk.CTkTextbox(self.left_frame)
        self.log_box.grid(row=21, column=0, columnspan=2, padx=20, pady=(10, 20), sticky="nsew")
        self.log_box.configure(state="disabled")

        # --- 오른쪽 메인 프레임 ---
        self.right_frame = ctk.CTkFrame(self, corner_radius=0)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.right_frame.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_columnconfigure(1, weight=1)

        self.image_list_frame = ctk.CTkScrollableFrame(self.right_frame, label_text="이미지 목록", width=250)
        self.image_list_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ns")
        self.image_buttons = []

        self.preview_canvas = ctk.CTkCanvas(self.right_frame, highlightthickness=0, bg="#2b2b2b")
        self.preview_canvas.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # --- 변수 초기화 ---
        self.image_dir, self.label_dir, self.output_dir = "", "", ""
        self.image_files, self.current_image_filename = [], None
        self.original_image, self.display_image_tk = None, None
        self.start_x_canvas, self.start_y_canvas = None, None
        self.pad_x, self.pad_y, self.scale_factor = 0, 0, 1.0
        self.rect_id = None
        self.classes = []
        self.class_colors = {}
        self.current_class_id = -1
        self.source_dir = ""

        self.preview_canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.preview_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.preview_canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.preview_canvas.bind("<Configure>", self.on_preview_resize)
        self.preview_canvas.bind("<Button-3>", self.on_right_click) # 우클릭 이벤트 바인딩

        self.load_classes()
        self.load_initial_dirs()

    def on_right_click(self, event):
        if self.original_image is None or not self.current_bboxes:
            return

        # 캔버스 좌표를 원본 이미지 좌표로 변환
        img_x = (event.x - self.pad_x) / self.scale_factor
        img_y = (event.y - self.pad_y) / self.scale_factor

        h, w, _ = self.original_image.shape
        
        # 클릭된 BBox 찾기 (가장 위에 있는 것부터)
        clicked_bbox_index = -1
        for i in reversed(range(len(self.current_bboxes))):
            bbox_data = self.current_bboxes[i]
            class_id, x_c, y_c, w_norm, h_norm = bbox_data
            
            x1 = (x_c - w_norm / 2) * w
            y1 = (y_c - h_norm / 2) * h
            x2 = (x_c + w_norm / 2) * w
            y2 = (y_c + h_norm / 2) * h

            if x1 <= img_x <= x2 and y1 <= img_y <= y2:
                clicked_bbox_index = i
                break
        
        if clicked_bbox_index != -1:
            if messagebox.askyesno("삭제 확인", "이 바운딩 박스를 삭제하시겠습니까?"):
                self.delete_bbox_at_index(clicked_bbox_index)

    def delete_bbox_at_index(self, index_to_delete):
        if not self.current_image_filename or not self.label_dir:
            return

        label_path = os.path.join(self.label_dir, os.path.splitext(self.current_image_filename)[0] + '.txt')

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            with open(label_path, 'w') as f:
                for i, line in enumerate(lines):
                    if i != index_to_delete:
                        f.write(line)
            
            class_id_to_delete = self.current_bboxes[index_to_delete][0]
            class_name = self.classes[int(class_id_to_delete)]
            self.log(f"BBox 삭제됨: {os.path.basename(label_path)}의 {index_to_delete+1}번째 라인 (클래스: {class_name})")
            
            # 프리뷰 새로고침
            self.show_preview(self.current_image_filename)
        else:
            self.log(f"오류: BBox를 삭제하려 했으나 레이블 파일을 찾을 수 없습니다: {label_path}")

    def select_source_dir(self):
        self.source_dir = self.select_folder("분할할 데이터셋 폴더를 선택하세요", self.source_dir_label)

    def split_dataset(self):
        if not self.source_dir:
            messagebox.showerror("오류", "먼저 분할할 데이터셋 폴더를 선택해주세요.")
            return

        images_dir = os.path.join(self.source_dir, 'images')
        labels_dir = os.path.join(self.source_dir, 'labels')

        if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
            messagebox.showerror("오류", f"선택한 폴더에 'images'와 'labels' 하위 폴더가 모두 존재해야 합니다.")
            return

        train_dir = os.path.join(self.source_dir, 'train')
        valid_dir = os.path.join(self.source_dir, 'valid')
        
        output_dirs = {
            "train_images": os.path.join(train_dir, 'images'),
            "train_labels": os.path.join(train_dir, 'labels'),
            "valid_images": os.path.join(valid_dir, 'images'),
            "valid_labels": os.path.join(valid_dir, 'labels'),
        }

        for d in output_dirs.values():
            os.makedirs(d, exist_ok=True)

        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(image_files)

        split_index = int(len(image_files) * 0.8)
        train_files = image_files[:split_index]
        valid_files = image_files[split_index:]

        self.log("데이터셋 분할을 시작합니다...")

        def copy_files(files, file_type):
            for filename in files:
                base_name, _ = os.path.splitext(filename)
                label_filename = base_name + '.txt'

                if file_type == 'train':
                    src_img_path = os.path.join(images_dir, filename)
                    dst_img_path = os.path.join(output_dirs["train_images"], filename)
                    src_lbl_path = os.path.join(labels_dir, label_filename)
                    dst_lbl_path = os.path.join(output_dirs["train_labels"], label_filename)
                else: # validation
                    src_img_path = os.path.join(images_dir, filename)
                    dst_img_path = os.path.join(output_dirs["valid_images"], filename)
                    src_lbl_path = os.path.join(labels_dir, label_filename)
                    dst_lbl_path = os.path.join(output_dirs["valid_labels"], label_filename)

                # Copy image
                if os.path.exists(src_img_path):
                    shutil.copy(src_img_path, dst_img_path)
                    self.log(f"{filename} -> {file_type} 폴더로 복사 완료.")
                
                # Copy label
                if os.path.exists(src_lbl_path):
                    shutil.copy(src_lbl_path, dst_lbl_path)

        copy_files(train_files, 'train')
        copy_files(valid_files, 'valid')

        self.log("데이터셋 분할 완료!")
        messagebox.showinfo("완료", "데이터셋 분할이 완료되었습니다.")


    def load_classes(self):
        try:
            with open("classes.txt", "r", encoding='utf-8') as f:
                self.classes = [line.strip() for line in f if line.strip()]
            if self.classes:
                # BGR format for OpenCV
                colors = [
                    (0, 0, 255),    # Red
                    (0, 165, 255),  # Orange
                    (0, 255, 255),  # Yellow
                    (0, 255, 0),    # Green
                    (255, 0, 0),    # Blue
                    (130, 0, 75),   # Indigo
                    (255, 0, 139),  # Violet
                ]
                self.class_colors = {i: colors[i % len(colors)] for i in range(len(self.classes))}
                self.class_menu.configure(values=self.classes)
                self.class_var.set(self.classes[0])
                self.current_class_id = 0
            else:
                self.class_menu.configure(values=["classes.txt 파일을 찾을 수 없거나 비어있습니다."])
        except FileNotFoundError:
            self.class_menu.configure(values=["classes.txt 파일을 찾을 수 없습니다."])

    def on_class_change(self, selected_class):
        if selected_class in self.classes:
            self.current_class_id = self.classes.index(selected_class)

    def on_preview_resize(self, event=None):
        if self.original_image is not None:
            base_image = self.draw_bboxes(self.original_image.copy())
            self.update_preview_image(base_image)

    def load_initial_dirs(self):
        initial_image_dir = r"C:\Users\SBA\github\training\images"
        initial_label_dir = r"C:\Users\SBA\github\training\labels"
        if os.path.isdir(initial_image_dir):
            self.image_dir = initial_image_dir
            self.image_dir_label.configure(text=initial_image_dir)
            self.load_image_list()
        if os.path.isdir(initial_label_dir):
            self.label_dir = initial_label_dir
            self.label_dir_label.configure(text=initial_label_dir)

    def select_folder(self, title, label_widget):
        root = Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title=title)
        root.destroy()
        if folder:
            label_widget.configure(text=folder)
        return folder

    def select_image_dir(self):
        self.image_dir = self.select_folder("이미지 폴더를 선택하세요", self.image_dir_label)
        if self.image_dir: self.load_image_list()

    def select_label_dir(self):
        self.label_dir = self.select_folder("YOLO 레이블(.txt) 폴더를 선택하세요", self.label_dir_label)

    def select_output_dir(self):
        self.output_dir = self.select_folder("증강된 파일을 저장할 출력 폴더를 선택하세요", self.output_dir_label)

    def load_image_list(self):
        if not self.image_dir or not os.path.isdir(self.image_dir): return
        for button in self.image_buttons: button.destroy()
        self.image_buttons.clear(); self.image_files.clear()
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        try: self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.lower().endswith(supported_formats)])
        except FileNotFoundError: messagebox.showerror("오류", f"폴더를 찾을 수 없습니다: {self.image_dir}"); return
        for filename in self.image_files:
            button = ctk.CTkButton(self.image_list_frame, text=filename, fg_color="transparent", command=lambda f=filename: self.show_preview(f))
            button.pack(fill="x", padx=5, pady=2)
            self.image_buttons.append(button)

    def show_preview(self, filename):
        self.current_image_filename = filename
        image_path = os.path.join(self.image_dir, filename)
        self.original_image = cv2.imread(image_path)
        if self.original_image is None: return

        # BBox 데이터 로드
        self.load_bboxes_for_current_image()
        
        self.on_preview_resize()

    def load_bboxes_for_current_image(self):
        self.current_bboxes = []
        if not self.label_dir or not self.current_image_filename:
            return
        
        label_path = os.path.join(self.label_dir, os.path.splitext(self.current_image_filename)[0] + '.txt')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    try:
                        parts = list(map(float, line.strip().split()))
                        if len(parts) == 5:
                            self.current_bboxes.append(parts)
                    except ValueError:
                        continue # Skip malformed lines

    def draw_bboxes(self, image):
        if not self.current_bboxes:
            return image
            
        h, w, _ = image.shape
        for bbox_data in self.current_bboxes:
            try:
                class_id, x_c, y_c, w_norm, h_norm = bbox_data
                x1 = int((x_c - w_norm / 2) * w)
                y1 = int((y_c - h_norm / 2) * h)
                x2 = int((x_c + w_norm / 2) * w)
                y2 = int((y_c + h_norm / 2) * h)
                class_id_int = int(class_id)
                
                if not (0 <= class_id_int < len(self.classes)):
                    continue

                color = self.class_colors.get(class_id_int, (0, 255, 0)) # Default to green
                class_name = self.classes[class_id_int]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            except (ValueError, IndexError):
                continue
        return image

    def update_preview_image(self, cv2_image):
        image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        widget_w = self.preview_canvas.winfo_width()
        widget_h = self.preview_canvas.winfo_height()

        if widget_w < 2 or widget_h < 2: return

        orig_w, orig_h = pil_image.size
        
        self.scale_factor = min(widget_w / orig_w, widget_h / orig_h)
        shrunk_w = int(orig_w * self.scale_factor)
        shrunk_h = int(orig_h * self.scale_factor)

        display_pil = pil_image.resize((shrunk_w, shrunk_h), Image.Resampling.LANCZOS)
        
        self.pad_x = (widget_w - shrunk_w) / 2
        self.pad_y = (widget_h - shrunk_h) / 2

        self.display_image_tk = ImageTk.PhotoImage(display_pil)

        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(self.pad_x, self.pad_y, anchor="nw", image=self.display_image_tk)

    def _event_to_shrunken_coords(self, event_x, event_y):
        img_x = event_x - self.pad_x
        img_y = event_y - self.pad_y
        return img_x, img_y

    def on_mouse_press(self, event):
        if self.original_image is None or self.current_class_id == -1: return
        self.start_x_canvas, self.start_y_canvas = event.x, event.y

    def on_mouse_drag(self, event):
        if self.start_x_canvas is None: return

        if self.rect_id:
            self.preview_canvas.delete(self.rect_id)
        
        bgr_color = self.class_colors.get(self.current_class_id, (0, 0, 255)) # Default to red
        hex_color = f"#{bgr_color[2]:02x}{bgr_color[1]:02x}{bgr_color[0]:02x}" # BGR to RGB for hex

        self.rect_id = self.preview_canvas.create_rectangle(
            self.start_x_canvas, self.start_y_canvas, event.x, event.y,
            outline=hex_color, width=2
        )

    def on_mouse_release(self, event):
        if self.start_x_canvas is None: return

        shrunk_x1, shrunk_y1 = self._event_to_shrunken_coords(self.start_x_canvas, self.start_y_canvas)
        shrunk_x2, shrunk_y2 = self._event_to_shrunken_coords(event.x, event.y)

        self.start_x_canvas, self.start_y_canvas = None, None
        if self.rect_id:
            self.preview_canvas.delete(self.rect_id)
            self.rect_id = None

        if self.current_class_id != -1:
            self.save_new_bbox(shrunk_x1, shrunk_y1, shrunk_x2, shrunk_y2, self.current_class_id)
        else:
            messagebox.showwarning("경고", "클래스를 먼저 선택해주세요.")
            self.show_preview(self.current_image_filename)

    def save_new_bbox(self, shrunk_x1, shrunk_y1, shrunk_x2, shrunk_y2, class_id):
        if self.scale_factor == 0: return
        
        orig_x1 = shrunk_x1 / self.scale_factor
        orig_y1 = shrunk_y1 / self.scale_factor
        orig_x2 = shrunk_x2 / self.scale_factor
        orig_y2 = shrunk_y2 / self.scale_factor

        img_h, img_w, _ = self.original_image.shape
        
        abs_x1, abs_x2 = min(orig_x1, orig_x2), max(orig_x1, orig_x2)
        abs_y1, abs_y2 = min(orig_y1, orig_y2), max(orig_y1, orig_y2)

        abs_x1, abs_y1 = max(0, abs_x1), max(0, abs_y1)
        abs_x2, abs_y2 = min(img_w - 1, abs_x2), min(img_h - 1, abs_y2)

        if abs_x1 >= abs_x2 or abs_y1 >= abs_y2: 
            self.on_preview_resize() # 잘못된 박스일 경우, 그냥 현재 상태로 다시 그립니다.
            return

        yolo_w = (abs_x2 - abs_x1) / img_w
        yolo_h = (abs_y2 - abs_y1) / img_h
        yolo_x_center = ((abs_x1 + abs_x2) / 2) / img_w
        yolo_y_center = ((abs_y1 + abs_y2) / 2) / img_h

        # 레이블 파일에 추가
        label_path = os.path.join(self.label_dir, os.path.splitext(self.current_image_filename)[0] + '.txt')
        line_to_write = f"{class_id} {yolo_x_center:.6f} {yolo_y_center:.6f} {yolo_w:.6f} {yolo_h:.6f}\n"
        with open(label_path, 'a') as f:
            f.write(line_to_write)
        
        # 메모리 내 리스트에 추가
        new_bbox_data = [float(p) for p in line_to_write.strip().split()]
        self.current_bboxes.append(new_bbox_data)
        
        self.log(f"새 BBox 저장: {os.path.basename(label_path)} - class {self.classes[class_id]}")
        
        # 화면 즉시 갱신
        self.on_preview_resize()

    def log(self, message):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")
        self.update_idletasks()

    def augment_and_save(self):
        if not all([self.image_dir, self.label_dir, self.output_dir]):
            messagebox.showerror("오류", "이미지, 레이블, 출력 폴더를 모두 선택해야 합니다.")
            return

        active_transforms = []
        if self.flip_h_var.get():
            active_transforms.append((A.HorizontalFlip(p=1.0), "hf"))
        if self.flip_v_var.get():
            active_transforms.append((A.VerticalFlip(p=1.0), "vf"))
        if self.rot_var.get():
            try:
                angle = float(self.rot_entry.get())
                active_transforms.append((A.SafeRotate(limit=(angle, angle), p=1.0, border_mode=cv2.BORDER_CONSTANT, fill=0), f"rot{angle}"))
            except ValueError:
                messagebox.showerror("오류", "회전 각도는 숫자로 입력해야 합니다.")
                return
        if self.scale_var.get():
            try:
                scale_limit = float(self.scale_entry.get())
                if not (0 < scale_limit < 2):
                    messagebox.showerror("오류", "크기 조절 비율은 0과 2 사이의 값이어야 합니다.")
                    return
                # RandomScale을 사용하되, 범위를 동일하게 설정하여 예측 가능한 결과 보장
                active_transforms.append((A.RandomScale(scale_limit=(scale_limit - 1.0, scale_limit - 1.0), p=1.0), f"scl{scale_limit}"))
            except ValueError:
                messagebox.showerror("오류", "크기 조절 비율은 숫자로 입력해야 합니다.")
                return

        if not active_transforms:
            messagebox.showinfo("정보", "적어도 하나 이상의 증강 옵션을 선택해주세요.")
            return

        base_output_dir = os.path.join(self.output_dir, "augmented")
        output_images_dir = os.path.join(base_output_dir, "images")
        output_labels_dir = os.path.join(base_output_dir, "labels")
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)

        self.log("증강 작업을 시작합니다...")
        total_files = len(self.image_files)
        for i, filename in enumerate(self.image_files):
            self.log(f"[{i+1}/{total_files}] 처리 중: {filename}")
            
            image_path = os.path.join(self.image_dir, filename)
            label_path = os.path.join(self.label_dir, os.path.splitext(filename)[0] + '.txt')

            image = cv2.imread(image_path)
            if image is None: 
                self.log(f"오류: {filename}을 읽을 수 없습니다.")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            bboxes = []
            class_labels = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_labels.append(parts[0])
                            bboxes.append([float(p) for p in parts[1:]])
            
            for transform, tag in active_transforms:
                try:
                    augmentation_pipeline = A.Compose(
                        [transform],
                        bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
                    )
                    augmented = augmentation_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
                    aug_image = augmented['image']
                    aug_bboxes = augmented['bboxes']

                    if not aug_bboxes:
                        continue

                    base_name, ext = os.path.splitext(filename)
                    new_image_filename = f"{base_name}_{tag}{ext}"
                    new_label_filename = f"{base_name}_{tag}.txt"

                    cv2.imwrite(os.path.join(output_images_dir, new_image_filename), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

                    with open(os.path.join(output_labels_dir, new_label_filename), 'w') as f:
                        for j, bbox in enumerate(aug_bboxes):
                            class_id = augmented['class_labels'][j]
                            bbox_str = ' '.join(map(lambda x: f"{x:.6f}", bbox))
                            f.write(f"{class_id} {bbox_str}\n")

                except Exception as e:
                    self.log(f"오류 발생: {filename} ({tag}) - {e}")
                    continue

        self.log("증강 작업 완료!")
        messagebox.showinfo("완료", f"{total_files}개 이미지의 증강이 완료되었습니다.")

if __name__ == "__main__":
    app = App()
    app.mainloop()
