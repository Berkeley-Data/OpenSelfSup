from ..registry import DATASOURCES
from .sen12ms_image_list import Sen12MSImageList


@DATASOURCES.register_module
class Sen12MS(Sen12MSImageList):

    def __init__(self, root, list_file, memcached, mclient_path, return_label=True, *args, **kwargs):
        super(Sen12MS, self).__init__(
            root, list_file, memcached, mclient_path, return_label)
