import struct
import numpy as np
import sys

BITMAP_FILE_HEADER_FMT = '<2sI4xI'
BITMAP_FILE_HEADER_SIZE = struct.calcsize(BITMAP_FILE_HEADER_FMT)
print(BITMAP_FILE_HEADER_SIZE)
BITMAP_INFO_FMT = '<I2i2H6I'
BITMAP_INFO_SIZE = struct.calcsize(BITMAP_INFO_FMT)

class BmpHeader:
    def __init__(self):
        self.bf_type = None
        self.bf_size = 0
        self.bf_off_bits = 0
        self.bi_size = 0
        self.bi_width = 0
        self.bi_height = 0
        self.bi_planes = 1  # 颜色平面数
        self.bi_bit_count = 0
        self.bi_compression = 0
        self.bi_size_image = 0
        self.bi_x_pels_per_meter = 0
        self.bi_y_pels_per_meter = 0
        self.bi_clr_used = 0
        self.bi_clr_important = 0


class BmpDecoder:
    def __init__(self, data):
        self.__header = BmpHeader()
        self.__data = data

    def read_header(self):
        if self.__header.bf_type is not None:
            return self.__header
        # bmp信息头
        self.__header.bf_type, self.__header.bf_size, \
        self.__header.bf_off_bits = struct.unpack_from(BITMAP_FILE_HEADER_FMT, self.__data)
        if self.__header.bf_type != b'BM':
            return None
        # 位图信息头
        self.__header.bi_size, self.__header.bi_width, self.__header.bi_height, self.__header.bi_planes, \
        self.__header.bi_bit_count, self.__header.bi_compression, self.__header.bi_size_image, \
        self.__header.bi_x_pels_per_meter, self.__header.bi_y_pels_per_meter, self.__header.bi_clr_used, \
        self.__header.bi_clr_important = struct.unpack_from(BITMAP_INFO_FMT, self.__data, BITMAP_FILE_HEADER_SIZE)
        return self.__header

    def read_data(self):
        header = self.read_header()
        if header is None:
            return None
        # 目前只写了解析常见的24位或32位位图
        if header.bi_bit_count != 24 and header.bi_bit_count != 32:
            return None
        # 目前只写了RGB模式
        if header.bi_compression != 0:
            return None
        offset = header.bf_off_bits
        channel = int(header.bi_bit_count / 8)
        img = np.zeros([header.bi_height, header.bi_width, channel], np.uint8)
        y_axis = range(header.bi_height - 1, -1, -1) if header.bi_height > 0 else range(0, header.bi_height)
        for y in y_axis:
            for x in range(0, header.bi_width):
                plex = np.array(struct.unpack_from('<' + str(channel) + 'B', self.__data, offset), np.int8)
                img[y][x] = plex
                offset += channel
        return img


class BmpEncoder:
    def __init__(self, img):
        self.__img = img

    def write_data(self):
        image_height, image_width, channel = self.__img.shape
        # 只支持RGB或者RGBA图片
        if channel != 3 and channel != 4:
            return False
        header = BmpHeader()
        header.bf_type = b'BM'
        header.bi_bit_count = channel * 8
        header.bi_width = image_width
        header.bi_height = image_height
        header.bi_size = BITMAP_INFO_SIZE
        header.bf_off_bits = header.bi_size + BITMAP_FILE_HEADER_SIZE
        header.bf_size = header.bf_off_bits + image_height * image_width * channel
        buffer = bytearray(header.bf_size)
        # bmp信息头
        struct.pack_into(BITMAP_FILE_HEADER_FMT, buffer, 0, header.bf_type, header.bf_size, header.bf_off_bits)
        # 位图信息头
        struct.pack_into(BITMAP_INFO_FMT, buffer, BITMAP_FILE_HEADER_SIZE, header.bi_size, header.bi_width,
                         header.bi_height,
                         header.bi_planes, header.bi_bit_count, header.bi_compression, header.bi_size_image,
                         header.bi_x_pels_per_meter, header.bi_y_pels_per_meter, header.bi_clr_used,
                         header.bi_clr_important)
        # 位图，一般都是纵坐标倒序模式
        offset = header.bf_off_bits
        for y in range(header.bi_height - 1, -1, -1):
            for x in range(header.bi_width):
                struct.pack_into('<' + str(channel) + 'B', buffer, offset, *self.__img[y][x])
                offset += channel
        return buffer