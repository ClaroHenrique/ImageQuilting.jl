# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

# CPU VERSION #
function preprocimagetoimfilter_cpu(img)
  img
end

function imfilter_cpu(img, krn)
  imfilter(img, centered(krn), Inner(), Algorithm.FFT())
end


# GPU VERSION #
function preprocimagetoimfilter_gpu(img)
  img |> CuArray |> CUFFT.fft
end

function imfilter_gpu(img, krn)
  imfilter_gpu(preprocimagetoimfilter_gpu(img), krn)
end

function imfilter_gpu(img::CuArray, krn)
  # retrieve basic info
  N = ndims(img)
  T = eltype(img)

  # pad kernel to common size with image
  padsize = size(img) .- size(krn)
  padkrn  = padarray(krn, Fill(zero(T), ntuple(i->0, N), padsize))

  # perform ifft(img .* conj.(fft(krn)))
  # img is already a CuArray
  fftimg = img |> CUFFT.fft
  fftkrn = padkrn |> CuArray |> CUFFT.fft
  result = (fftimg .* conj.(fftkrn)) |> CUFFT.ifft

  # recover result
  finalsize = size(img) .- (size(krn) .- 1)
  real.(result[CartesianIndices(finalsize)]) |> Array
end

const preprocimagetoimfilter_kernel = CUDA.functional() ? preprocimagetoimfilter_gpu : preprocimagetoimfilter_cpu
const imfilter_kernel = CUDA.functional() ? imfilter_gpu : imfilter_cpu