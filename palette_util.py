import numpy as np
import cv2

class PaletteUtil:
    def __init__(self):
        self.palettes = {}
        try:
            self.palettes["ARCTIC"] = np.loadtxt("./assets/ARCTIC.csv", delimiter=",", dtype=np.uint8)
            self.palettes["INFER"] = np.loadtxt("./assets/INFER.csv", delimiter=",", dtype=np.uint8)
            self.palettes["IRON"] = np.loadtxt("./assets/IRON.csv", delimiter=",", dtype=np.uint8)
            self.palettes["RAINBOW"] = np.loadtxt("./assets/RAINBOW.csv", delimiter=",", dtype=np.uint8)
            self.palettes["REDGRAY"] = np.loadtxt("./assets/REDGRAY.csv", delimiter=",", dtype=np.uint8)
        except Exception as e:
            print("Palette File Loading Error:", e)

    def normalize_image(self, image):
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val == min_val:
            return np.zeros_like(image, dtype=np.uint8)
        return ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    def apply_palette_lut(self, normalized_img, palette_lut):
        # 2D(그레이스케일)면 3채널로 변환
        if len(normalized_img.shape) == 2 or (len(normalized_img.shape) == 3 and normalized_img.shape[2] == 1):
            normalized_img_3ch = cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2BGR)
        else:
            normalized_img_3ch = normalized_img

        # 채널별로 LUT 적용
        b = cv2.LUT(normalized_img_3ch[:, :, 0], palette_lut[:, 0])
        g = cv2.LUT(normalized_img_3ch[:, :, 1], palette_lut[:, 1])
        r = cv2.LUT(normalized_img_3ch[:, :, 2], palette_lut[:, 2])
        return cv2.merge([b, g, r])

    def resize_and_sharpen(self, image, out_size=(640, 480)):
        image_resized = cv2.resize(image, out_size, interpolation=cv2.INTER_LINEAR)
        sharpen_kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
        return cv2.filter2D(image_resized, -1, sharpen_kernel)

    def apply_color_palette(self, image, palette_name, out_size=(640, 480)):
        if palette_name == "DEFAULT" or palette_name not in self.palettes:
            return None
        try:
            normalized_img = self.normalize_image(image)
            palette_lut = self.palettes[palette_name][:, [2, 1, 0]].astype(np.uint8)
            palette_lut = np.ascontiguousarray(palette_lut)
            color_mapped_img = self.apply_palette_lut(normalized_img, palette_lut)
            result_img = self.resize_and_sharpen(color_mapped_img, out_size)
            return result_img
        except Exception as e:
            print(f"LUT Error: {e}")
            return None