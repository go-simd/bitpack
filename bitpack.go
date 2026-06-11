// Package bitpack implements the FastPFOR/simdcomp bit-packing primitive: pack
// blocks of 128 uint32 values, each using exactly `bits` bits (1..32), into a
// tight little-endian bitstream, and the inverse unpack.
//
// Layout. A block of 128 integers is packed into bits*16 bytes, viewed as
// `bits` consecutive 128-bit words. The block is organised as 4 interleaved
// lanes of 32 integers: lane j (j=0..3) carries the substream
//
//	src[j], src[4+j], src[8+j], ..., src[124+j]
//
// packed contiguously, low integer in the low bits, into the j-th uint32
// column of the output words. This is exactly the SSE layout used by Lemire's
// simdcomp, so the packed bytes are identical to that C library's
// simdpack/SIMD_fastpackwithoutmask output. On amd64 the bulk is done by a
// generated SIMD kernel (SSE2, or AVX2 when available); other arches use the
// scalar reference. Output is byte-identical regardless of path.
package bitpack

// BlockSize is the number of uint32 values per packed block.
const BlockSize = 128

// PackedLen returns the number of bytes that Pack writes for src, which must be
// a whole number of 128-int blocks. It equals 16*bits per block.
func PackedLen(nInts, bits int) int { return nInts / BlockSize * bits * 16 }

// Pack packs src (length a multiple of 128) into dst using `bits` bits per
// value (1..32) and returns the number of bytes written, 16*bits per block.
// Only the low `bits` bits of each src value are used; higher bits are ignored
// (masked), matching simdcomp's fastpackwithoutmask semantics. dst must hold at
// least PackedLen(len(src), bits) bytes.
func Pack(dst []byte, src []uint32, bits int) int {
	if bits < 1 || bits > 32 {
		panic("bitpack: bits out of range [1,32]")
	}
	if len(src)%BlockSize != 0 {
		panic("bitpack: src length not a multiple of 128")
	}
	n := PackedLen(len(src), bits)
	if n == 0 {
		return 0
	}
	_ = dst[n-1] // bounds check hint
	packBulk(dst, src, bits, len(src)/BlockSize)
	return n
}

// Unpack is the inverse of Pack: it reads len(dst)/128 blocks from src (16*bits
// bytes each) and writes the recovered values into dst (length a multiple of
// 128). Each output value is in [0, 2^bits).
func Unpack(dst []uint32, src []byte, bits int) {
	if bits < 1 || bits > 32 {
		panic("bitpack: bits out of range [1,32]")
	}
	if len(dst)%BlockSize != 0 {
		panic("bitpack: dst length not a multiple of 128")
	}
	blocks := len(dst) / BlockSize
	if blocks == 0 {
		return
	}
	_ = src[blocks*16*bits-1] // bounds check hint
	unpackBulk(dst, src, bits, blocks)
}
