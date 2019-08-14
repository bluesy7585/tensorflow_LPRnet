import argparse
import os
import random
import sys
import cv2
import numpy as np

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from data_aug import data_augmentation
import model.LPRnet as model

class ImageGenerator:
    def __init__(self, ttf_dir, char_set, char_height=36):

        self.chars = char_set
        self.letters = []
        self.digits = []
        for c in char_set:
            if str.isalpha(c):
                self.letters.append(c)
            else:
                self.digits.append(c)

        self.char_height = char_height
        self.ttf_dir = ttf_dir
        self.fonts, self.font_char_ims = self.load_fonts(ttf_dir)

        white = [1, 1, 1]
        yellow = [0, 1, 1]
        blue = [1, 0, 0]
        self.black_text_colors = [white, yellow]
        self.white_text_colors = [blue]

    def random_text_plate_colors(self, min_diff=0.3, black_text=True):
        high = random.uniform(min_diff, 1.0)
        low = random.uniform(0.0, high - min_diff)
        text_color, plate_color = (low, high) if black_text else (high, low)
        return text_color, plate_color

    def load_fonts(self, folder_path):
        font_char_ims = {}
        fonts = [f for f in os.listdir(folder_path) if f.endswith('.ttf')]
        for font in fonts:
            font_char_ims[font] = dict(self.generate_char_imgs(\
                os.path.join(folder_path, font), self.char_height))
        return fonts, font_char_ims

    def generate_char_imgs(self, font_path, output_height):
        font_size = output_height * 4
        font = ImageFont.truetype(font_path, font_size)
        height = max(font.getsize(c)[1] for c in self.chars)

        for c in self.chars:
            width = font.getsize(c)[0]
            im = Image.new("RGBA", (width, height), (0, 0, 0))

            draw = ImageDraw.Draw(im)
            draw.text((0, 0), c, (255, 255, 255), font=font)
            scale = float(output_height) / height
            im = im.resize((int(width * scale), output_height), Image.ANTIALIAS)
            yield c, np.array(im)[:, :, 0]

    def generate_code(self):
        pre_n = random.randint(2, 3)
        pre_letters = [random.choice(self.letters) for _ in range(pre_n)]
        digit_n = random.randint(2, 3)
        digits = [random.choice(self.digits) for _ in range(digit_n)]
        post_n = random.randint(2, 4)
        post_letters = [random.choice(self.letters) for _ in range(post_n)]

        code = ''.join(pre_letters) + ''.join(digits) + '-' + ''.join(post_letters)
        return code

    def getOneRandomFont(self):
        return random.choice(self.fonts)

    def getCharGivenLabelFont(self, label, font):
        char_ims = self.font_char_ims[font]
        char_img = char_ims[label]
        return char_img, label

    def generate_images(self, number):

        images = []
        labels = []

        for _ in enumerate(range(number)):

            char_height = self.char_height
            code = self.generate_code()

            space = round(char_height * random.uniform(0.0, 0.3))
            char_spacing = []
            for c in code:
                if c == '-':
                    char_spacing[-1] += space
                else:
                    char_spacing.append(space)

            code = code.replace('-','')

            # generate letter, number images
            char_ims = []
            char_font = self.getOneRandomFont()

            for i, c in enumerate(code):
                char, label = self.getCharGivenLabelFont(c, char_font)
                char_ims.append(char)

            char_width_sum = sum(char_im.shape[1] for char_im in char_ims)

            top_padding = round(random.uniform(0.1, 1.0) * char_height)
            bot_padding = round(random.uniform(0.1, 1.0) * char_height)
            left_padding = round(random.uniform(0.1, 2.0) * char_height)
            right_padding = round(random.uniform(0.1, 2.0) * char_height)

            Plate_h = (char_height + top_padding + bot_padding)
            Plate_w = (char_width_sum + left_padding + right_padding + sum(char_spacing[:-1]))

            out_shape = (Plate_h, Plate_w)
            text_mask = np.zeros(out_shape)

            x = left_padding
            y = top_padding

            for ind, c in enumerate(code):
                char_im = char_ims[ind]
                ix, iy = int(x), int(y)
                text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
                x += char_im.shape[1] + char_spacing[ind]

            is_black_text = random.choice([True, False])
            text_color, plate_color = self.random_text_plate_colors(black_text=is_black_text)

            plate_mask = (255. - text_mask)

            if is_black_text:
                color = np.array(random.choice(self.black_text_colors))
            else:
                color = np.array(random.choice(self.white_text_colors))

            w_color = color * plate_color

            dim = (Plate_h, Plate_w, 3)
            Plate = np.ones(dim)
            Plate[:, :, 0] = text_mask * text_color
            Plate[:, :, 1] = text_mask * text_color
            Plate[:, :, 2] = text_mask * text_color
            Plate[:, :, 0] += plate_mask * w_color[0]
            Plate[:, :, 1] += plate_mask * w_color[1]
            Plate[:, :, 2] += plate_mask * w_color[2]
            Plate = Plate.astype(np.uint8)

            images.append(Plate)
            labels.append(code)

        return images, labels



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save_dir", help="save directory",
                    type=str, default="./train")
    parser.add_argument("-n", "--num", help="number of images",
                    type=int, default=1000)

    args = parser.parse_args()
    save_dir = args.save_dir

    def labelToSaveFilename(label):
        rand_tail = random.randint(10000, 99999)
        name = '{}_{}.jpg'.format(label, rand_tail)
        return name

    FONT_HEIGHT = 32
    ttfCharGen = ImageGenerator('./fonts/', char_set=model.CHARS, char_height=FONT_HEIGHT)

    plates, labels = ttfCharGen.generate_images(args.num)
    #for plate in plates:
        #cv2.imshow('', plate)
        #cv2.waitKey(0)

    if not os.path.isdir(save_dir): os.mkdir(save_dir)

    for img, label in zip(plates, labels):
        full_path = os.path.join(save_dir, labelToSaveFilename(label))
        img = data_augmentation(img)
        cv2.imwrite(full_path, img)