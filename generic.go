//go:build !amd64

package bitpack

// On non-amd64 arches there is no SIMD kernel; the scalar reference does all the
// work (and is the correctness oracle the amd64 kernels are validated against).
func packBulk(dst []byte, src []uint32, bits, blocks int) {
	for b := 0; b < blocks; b++ {
		packBlockScalar(dst[b*16*bits:], src[b*BlockSize:], bits)
	}
}

func unpackBulk(dst []uint32, src []byte, bits, blocks int) {
	for b := 0; b < blocks; b++ {
		unpackBlockScalar(dst[b*BlockSize:], src[b*16*bits:], bits)
	}
}
