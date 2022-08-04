# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

function imfilter_cpu(img, krn)
  imfilter(img, centered(krn), Inner(), Algorithm.FFT())
end

function imfilter_gpu(img, krn)
  img = img |> collect |> Array
  krn = krn |> collect |> Array

  # retrieve basic info
  N = ndims(img)
  T = eltype(img)

  # pad kernel to common size
  padsize = size(img) .- size(krn)
  padkrn = padarray(krn, Fill(zero(T), ntuple(i->0, N), padsize))

  # perform ifft(fft(img) .* conj.(fft(krn)))
  fftimg = img |> CuArray |> CUFFT.fft
  fftkrn = padkrn |> CuArray |> CUFFT.fft
  result = (fftimg .* conj.(fftkrn)) |> CUFFT.ifft
  result_cpu = imfilter_cpu(img, krn) |> collect

  # recover result
  finalsize = size(img) .- (size(krn) .- 1)
  result = real.(result[CartesianIndices(finalsize)]) |> Array
  println(size(result), size(result_cpu), size(padkrn))
  println("testeeeeeeeeeeeeeeeeeeeeeeee!!!!!! - - - !!!!!!")
  println(all(result .≈ result_cpu))

  result
end

const imfilter_kernel = CUDA.functional() ? imfilter_gpu : imfilter_cpu