//go:build s390x

package bitpack

// On s390x the bulk of Pack/Unpack runs generated per-width vector-facility
// kernels (pack_s390x.s). For each width 1..32 there is a kernel processing one
// 128-int block per iteration. The vector facility is baseline for the s390x
// builds Go targets (z13+), so there is no runtime feature check: the build tag
// is the gate. s390x is big-endian; the kernels byte-reverse each 32-bit word
// in-register (VPERM) so the packed bytes stay byte-identical to the
// little-endian simdcomp stream -- see pack_s390x_gen.go. The Go prototypes and
// dispatch tables are in pack_decl_s390x.go (generated).
//
//go:generate go run pack_s390x_gen.go unpack_s390x_gen.go
//go:generate go mod edit -droprequire github.com/go-asmgen/asmgen
//go:generate go mod tidy

func packBulk(dst []byte, src []uint32, bits, blocks int) {
	packVXtab[bits](dst, src, blocks)
}

func unpackBulk(dst []uint32, src []byte, bits, blocks int) {
	unpackVXtab[bits](dst, src, blocks)
}

// packBlock/unpackBlock satisfy the single-block contract used by tests.
func packBlock(dst []byte, src []uint32, bits int)   { packVXtab[bits](dst, src, 1) }
func unpackBlock(dst []uint32, src []byte, bits int) { unpackVXtab[bits](dst, src, 1) }
