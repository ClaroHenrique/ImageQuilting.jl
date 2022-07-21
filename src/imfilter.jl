# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

krn_allocated_mem = nothing

function load_imfilter_img_to_cpu(img, krnsize)
  img
end

function imfilter_cpu(img, krn, imgsize)
  imfilter(img, centered(krn), Inner(), Algorithm.FFT())
end

function load_imfilter_img_to_gpu(img, krnsize)
  N = ndims(img)
  T = eltype(img)

  # pad img to common size with krn
  padimg = padarray(img, Fill(zero(T), ntuple(i->0, N), krnsize .- 1))
  fftimg = padimg |> CuArray |> CUFFT.fft
  fftimg
end


function load_imfilter_krn_to_gpu(krn, imgsize, krn_allocated_mem)
  N = ndims(krn)
  T = eltype(krn)

  copyto!(krn_allocated_mem, krn)
  krn_allocated_mem
end

function imfilter_gpu(fftimg, krn, imgsize, krn_allocated_mem)
  # load krn to gpu
  fftkrn = load_imfilter_krn_to_gpu(krn, imgsize, krn_allocated_mem)

  # perform ifft(fft(img) .* conj.(fft(krn)))
  result = (fftimg .* conj.(fftkrn)) |> CUFFT.ifft

  # unpad result
  N = ndims(krn)
  krnsize = size(krn)
  start  = CartesianIndex(ntuple(i->1, N))
  finish = CartesianIndex(imgsize .- (krnsize .- 1))
  real.(result[start:finish]) |> Array
end

const load_imfilter_img_to_kernel = CUDA.functional() ? load_imfilter_img_to_gpu : load_imfilter_img_to_cpu
const imfilter_kernel = CUDA.functional() ? imfilter_gpu : imfilter_cpu
