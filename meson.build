project('kmscube', ['c'], version : '0.1', license : 'MIT',
        default_options : ['c_std=gnu99'])

c = meson.get_compiler('c')

dep_libdrm = dependency('libdrm')
dep_gbm = dependency('gbm')
dep_egl = dependency('egl')
dep_glesv2 = dependency('glesv2')
dep_m = c.find_library('m', required : false)

kmscube = executable(
  'amazing-scroller-coaster',
  files('amazing-scroller-coaster.c'),
  c_args : ['-Wall',
            '-Werror=implicit-function-declaration',
	    '-Werror=missing-prototypes',
	    '-march=core-avx2',
	    '-D_DEFAULT_SOURCE'],
  dependencies : [dep_libdrm, dep_gbm, dep_egl, dep_glesv2, dep_m],
)
