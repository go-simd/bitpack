//go:build ppc64le

package bitpack

// On ppc64le the bulk of Pack/Unpack runs generated per-width VSX kernels
// (pack_ppc64le.s). For each width 1..32 there is a kernel processing one
// 128-int block per iteration. VSX is baseline on every ppc64le CPU Go targets
// (POWER8+), so there is no runtime feature check: the build tag is the gate.
// The Go prototypes and dispatch tables are in pack_decl_ppc64le.go (generated).
//
//go:generate go run pack_ppc64le_gen.go unpack_ppc64le_gen.go
//go:generate go mod edit -droprequire github.com/go-asmgen/asmgen
//go:generate go mod tidy

func packBulk(dst []byte, src []uint32, bits, blocks int) {
	packVSXtab[bits](dst, src, blocks)
}

func unpackBulk(dst []uint32, src []byte, bits, blocks int) {
	unpackVSXtab[bits](dst, src, blocks)
}

// packBlock/unpackBlock satisfy the single-block contract used by tests.
func packBlock(dst []byte, src []uint32, bits int)   { packVSXtab[bits](dst, src, 1) }
func unpackBlock(dst []uint32, src []byte, bits int) { unpackVSXtab[bits](dst, src, 1) }
