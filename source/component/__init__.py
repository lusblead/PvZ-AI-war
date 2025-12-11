# source/component/__init__.py
from . import menubar
from . import map
from . import plant
from . import zombie

# 方便直接导入类
from .menubar import MenuBar, MoveBar, Panel
from .plant import *
from .zombie import *
from .map import Map
