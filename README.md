ImageQuilting.jl
================

Image quilting for texture synthesis in Julia.

Dependencies
------------

* [Images](https://github.com/timholy/Images.jl)
* [ImageView](https://github.com/timholy/ImageView.jl) (Optional)

Usage
-----

```julia
synthesis = imquilt(img::Image, tilesize, n; tol=1e-3)
synthesis = imquilt(img::AbstractArray, tilesize, n; tol=1e-3)
```

where:

* `img` can be any 2D (RGB or Grayscale) image
* `tilesize` is the tile size used to scan `img`
* `n` is the number of tiles to stitch together in the output
* `tol` is the tolerance used for finding best tiles

Example
-------

Reproduce some of the paper results with:

```bash
julia example.jl
```

REFERENCES
----------

EFROS, A.; FREEMAN, W. T., 2001. Image Quilting for Texture Synthesis and Transfer. [DOWNLOAD](http://graphics.cs.cmu.edu/people/efros/research/quilting.html)
