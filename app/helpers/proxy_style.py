from PySide6.QtWidgets import QProxyStyle, QStyle, QStyleOptionSlider
from PySide6.QtCore import Qt

class ProxyStyle(QProxyStyle):
    """
    ProxyStyle que permite clicar na barra do slider para posicionar o handle (igual YouTube).
    """
    def styleHint(self, hint, option=None, widget=None, returnData=None):
        if hint == QStyle.SH_Slider_AbsoluteSetButtons and isinstance(option, QStyleOptionSlider):
            return Qt.LeftButton
        return super().styleHint(hint, option, widget, returnData)
