from module_filter import ThreeBandEqualizer
from module_ui import ThreeBandEqualizerGUI

if __name__ == "__main__":
    eq = ThreeBandEqualizer()
    gui = ThreeBandEqualizerGUI(eq)
    gui.run()