imgaug 库已经停止维护，可以使用 albumentations 库作为替代（推荐）。

如果使用imgaug库，出现以下报错信息：
    AttributeError: module 'numpy' has no attribute 'bool'.
    报错位置： File "D:\pythonx.x.xx\lib\site-packages\imgaug\augmenters\meta.py", line 3368, in _get_augmenter_active
    augmenter_active = np.zeros((nb_rows, len(self)), dtype=np.bool)

可以尝试以下解决方法：
    1. 将 numpy 版本降低到 1.19.x。
    2. 修改 imgaug 库源码，将报错位置3368行方法中的dtype=np.bool改为dtype=bool。
    3. 使用其他替代库，如albumentations。