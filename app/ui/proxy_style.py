from PySide6.QtWidgets import QProxyStyle, QStyle, QStyleOptionSlider

class ProxyStyle(QProxyStyle):
    """Permite que clicar na barra do slider posicione o handle diretamente."""

    def styleHint(self, hint, option=None, widget=None, returnData=None):
        # Esse override faz com que o clique na barra do slider mova o handle pro local clicado (comportamento mais moderno).
        if hint == QStyle.SH_Slider_AbsoluteSetButtons and isinstance(option, QStyleOptionSlider):
            return Qt.LeftButton
        return super().styleHint(hint, option, widget, returnData)
