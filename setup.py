from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

if __name__ == "__main__":
  setup( 
    name = "humble_track",
    version = "0.0.0",
    package_dir = { "": "python" },
    cmdclass = { "build_ext": build_ext },
    ext_modules = [
      Extension( 
        "humble_track.integral_image", 
        ["python/humble_track/integral_image.pyx"],
        extra_compile_args = ["-O3"]
      ),
      Extension( 
        "humble_track.compute_texcels", 
        ["python/humble_track/compute_texcels.pyx"],
        extra_compile_args = ["-O3"] 
      ),
      Extension( 
        "humble_track.find_groups", 
        ["python/humble_track/find_groups.pyx"],
        extra_compile_args = ["-O3"] 
      ),
      Extension( 
        "humble_track.compute_histograms", 
        ["python/humble_track/compute_histograms.pyx"],
        extra_compile_args = ["-O3"] 
      ),
      Extension( 
        "humble_track.foreground_groups", 
        ["python/humble_track/foreground_groups.pyx"],
        extra_compile_args = ["-O3"] 
      ),
      Extension( 
        "humble_track.kmeans_cpu", 
        ["python/humble_track/kmeans_cpu.pyx"],
        extra_compile_args = ["-O3"] 
      )
    ]
  )
  
