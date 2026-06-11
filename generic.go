//go:build !amd64 && !ppc64le && !s390x

package bitpack

// On arches without a SIMD kernel the scalar reference does all the work (and is
// the correctness oracle the SIMD kernels on amd64/ppc64le/s390x are validated
// against).
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
