import platform
from pathlib import Path

import os
import sys
import platform


# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

_PREFIX_PATH = Path(os.environ['VIPER_ROOT'])


__version__ = "1.2.0"


ext_modules = [
    Pybind11Extension(
        "piper_phonemize_cpp",
        [
            "src/python.cpp",
            "src/phonemize.cpp",
            "src/phoneme_ids.cpp",
            "src/tashkeel.cpp",
        ],
        define_macros=[("VERSION_INFO", __version__)],
        include_dirs=[str(_PREFIX_PATH / "include")],
        library_dirs=[str(_PREFIX_PATH / "lib")],
        libraries=["espeak-ng", "ucd", "speechPlayer", "onnxruntime"],
        extra_link_args=['-Wl,-rpath,' + str(_PREFIX_PATH / "lib")],
    ),
]

setup(
    name="piper_phonemize",
    version=__version__,
    author="ShadedBandit",
    author_email="animacion@greentwip.xyz",
    url="https://github.com/Cybertwip/viper",
    description="Phonemization libary used by Viper text to speech system",
    long_description="",
    packages=["piper_phonemize"],
    package_data={
        "piper_phonemize": [
            str(p) for p in (_PREFIX_PATH / "share" / "espeak-ng-data").rglob("*")
        ]
        + [str(_PREFIX_PATH / "share" / "libtashkeel_model.ort")]
    },
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)
