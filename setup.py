from distutils.core import setup

setup(
    name = 'byspec',
    descriptoin = 'BFOSC and YFOSC Spectra Data Reduction Pipeline',
    author      = 'Deyang Song, Liang Wang',
    license     = 'Apache-2.0',
    zip_safe    = False,
    packages    = ['byspec',
                   ],
    package_data = {
        'byspec': ['data/linelist/*',
                   ]

        },
    )
