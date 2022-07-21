from pathlib import Path, PosixPath
from typing import List


def find_item(items: List[Path], name: str, parent=None):
    if parent is not None:
        items = [item for item in items if item.parent.stem == parent]

    items = [item for item in items if item.name == name]
    if len(items):
        return items[0].resolve().as_posix()
    else:
        print(f"{name} not found")
        return None


class AssetMeta:
    def __init__(self):
        base_path = Path(__file__).parent
        self.haar_files = list(Path(f'{base_path}/haarcascade-xml').glob('*'))
        self.images = list(Path(f'{base_path}/IMAGES').rglob('*'))
        self.videos = list(Path(f'{base_path}/VIDEOS').rglob('*'))
        self.yolov3_files = list(Path(f'{base_path}/yolov3').rglob('*'))
        self.yolov4_files = list(Path(f'{base_path}/yolov4').rglob('*'))
        self.fontt = Path(f'{base_path}/Yekan.ttf').resolve().as_posix()
        self.classes = Path(f'{base_path}/yolov3/coco.names').resolve()
        self._init_haar()
        self._init_yolov3()
        self._init_yolov4()

    def _init_haar(self):
        self.car_xml = [i for i in self.haar_files if i.as_posix().endswith('cars.xml')][0].resolve()
        self.pedestrian_xml = [i for i in self.haar_files if i.as_posix().endswith('pedestrian.xml')][0].resolve()

    def _init_yolov3(self):
        self.yolov3_cfg = [i for i in self.yolov3_files if i.as_posix().endswith('config-yolov3.cfg')][0].resolve()
        self.yolov3_weights = [i for i in self.yolov3_files if i.as_posix().endswith('yolov3.weights')][0].resolve()

        self.yolov3_tiny_cfg = [i for i in self.yolov3_files if i.as_posix().endswith('config-tiny.cfg')][0].resolve()
        self.yolov3_tiny_weights = [i for i in self.yolov3_files if i.as_posix().endswith('yolov3-tiny.weights')][
            0].resolve()

    def _init_yolov4(self):
        self.yolov4_cfg = [i for i in self.yolov4_files if i.as_posix().endswith('yolov4.cfg')][0].resolve()
        self.yolov4_weights = [i for i in self.yolov4_files if i.as_posix().endswith('yolov4.weights')][0].resolve()

    def _init_videos(self):
        self.car_xml = [i for i in self.haar_files if i.as_posix().endswith('cars.xml')][0].resolve()

    @property
    def IMG_CHESS(self):
        ch1 = self.find_item(self.images, name='chess1.png', parent='corner')
        ch2 = self.find_item(self.images, name='chess2.jpg', parent='corner')
        return ch1, ch2

    @property
    def IMG_FACES(self):
        im1 = self.find_item(self.images, name='multi_face1.jpg', parent='face')
        im2 = self.find_item(self.images, name='multi_face_volleyb.jpg', parent='face')
        return im1, im2

    @property
    def IMG_FACE(self):
        im1 = self.find_item(self.images, name='face.jpeg', parent='face')
        im2 = self.find_item(self.images, name='2face.png', parent='face')
        return im1, im2

    @property
    def IMG_FACE_ON(self):
        im1 = self.find_item(self.images, name='glasses_1.png', parent='face_add_on')
        im2 = self.find_item(self.images, name='glasses_2.png', parent='face_add_on')
        return im1, im2

    @property
    def PPT(self):
        im1 = self.find_item(self.images, name='book.jpeg', parent='perspective_transform')
        im2 = self.find_item(self.images, name='book_2.png', parent='perspective_transform')
        return im1, im2

    @property
    def IMG_NORMAL(self):
        ave = self.find_item(self.images, name='aventador.png', parent='normal')
        boat = self.find_item(self.images, name='woman-boat.jpg', parent='normal')
        return ave, boat

    @staticmethod
    def find_item(items: List[Path], name: str, parent: str = None) -> str:
        if parent is not None:
            items = [item for item in items if item.parent.stem == parent]

        items = [item for item in items if item.name == name]
        if len(items):
            return items[0].resolve().as_posix()
        else:
            print(f"{name} not found")
            return ""

    @property
    def VIDEO_TT(self):
        video = find_item(self.videos, name='ping-pong.mp4', parent='optflow')
        return video

    @property
    def VIDEO_CARS_1(self):
        return find_item(self.videos, name='cars.avi', parent='tracking')

    @property
    def VIDEO_CARS_2(self):
        return find_item(self.videos, name='vehicles.mp4', parent='tracking')

    @property
    def VIDEO_PEDESTRIAN(self):
        return find_item(self.videos, name='pedestrians.avi', parent='tracking')

    @property
    def YOLOV3(self):
        return self.yolov3_cfg.as_posix(), self.yolov3_weights.as_posix(), self.classes

    @property
    def YOLOV3_TINY(self):
        return self.yolov3_tiny_cfg.as_posix(), self.yolov3_tiny_weights.as_posix(), self.classes

    @property
    def YOLOV4(self):
        return self.yolov4_cfg.as_posix(), self.yolov4_weights.as_posix(), self.classes

    @property
    def HAAR_XML(self):
        """
        Returns car xml file and pedestrian xml file.

        """
        return self.car_xml, self.pedestrian_xml

    @property
    def CUSTOM_FONT(self):
        return self.fontt


if __name__ == '__main__':
    m = AssetMeta()
    print(m.IMG_CHESS)
    print(m.IMG_FACE)
    print(m.IMG_FACES)
    print(m.IMG_NORMAL)
    print(m.HAAR_XML)
    print(m.YOLOV4)
    print(m.YOLOV3)
    print(m.YOLOV3_TINY)
    print(m.CUSTOM_FONT)
