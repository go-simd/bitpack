# bitpack

The **FastPFOR / simdcomp** bit-packing primitive in pure Go (CGO=0, stable Go,
no GOEXPERIMENT): pack blocks of **128 `uint32`** values, each using exactly
`bits` bits (1..32), into a tight little-endian bitstream — and the inverse.

```go
n := bitpack.Pack(dst, src, bits)   // src len a multiple of 128; n = 16*bits per block
bitpack.Unpack(out, dst, bits)
```

Output is **byte-identical to Lemire's simdcomp** (`SIMD_fastpackwithoutmask`):
the 128 integers are stored as 4 interleaved 32-int lanes. On amd64 the bulk runs
a generated **SSE2 / AVX2** kernel (one per bit-width, dispatched via
`x/sys/cpu`); other arches use the scalar reference. Every path is byte-exact —
verified by table tests, a scalar-equality test, and `FuzzPack`/`FuzzUnpack`.

## Performance

Native amd64 (ubuntu-latest CI, AVX2, `-count=6`), throughput over the input
`uint32` stream, **vs the package's own scalar reference**:

| op | bits=8 | bits=16 | bits=24 |
|---|---|---|---|
| **Pack** SIMD | ~62.0 GB/s | ~57.5 GB/s | ~46.8 GB/s |
| Pack scalar | ~2.31 GB/s | ~1.92 GB/s | ~1.46 GB/s |
| **Unpack** SIMD | ~44.5 GB/s | ~42.5 GB/s | ~40.4 GB/s |
| Unpack scalar | ~2.08 GB/s | ~1.85 GB/s | ~1.38 GB/s |

That's **~21–32×** over scalar — bit-packing is a textbook SIMD-friendly problem
(shifts + masks + ORs, no data-dependent control flow). go-asmgen generates the
per-width kernels.

> Competitor note: `ronanh/intcomp` is the prior pure-Go SIMD-ish integer codec
> for this space; a head-to-head benchmark against it is a follow-up.

## License

BSD-3-Clause.
