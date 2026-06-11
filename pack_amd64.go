//go:build amd64

package bitpack

import "golang.org/x/sys/cpu"

// The bulk of Pack/Unpack on amd64 runs generated per-width SIMD kernels
// (pack_amd64.s). For each width 1..32 there is an SSE kernel (one 128-int
// block per iteration) and an AVX2 kernel (two blocks per iteration). The Go
// prototypes and dispatch tables are in pack_decl_amd64.go (generated).
//
//go:generate go run pack_gen.go unpack_gen.go
//go:generate go mod edit -droprequire github.com/go-asmgen/asmgen
//go:generate go mod tidy

var hasAVX2 = cpu.X86.HasAVX2

// packBulk packs `blocks` 128-int blocks. When AVX2 is present it packs blocks
// in pairs (two per call) and finishes a trailing odd block with SSE.
func packBulk(dst []byte, src []uint32, bits, blocks int) {
	wb := 16 * bits // bytes per packed block
	if hasAVX2 && blocks >= 2 {
		pairs := blocks / 2
		packAVX2tab[bits](dst, src, pairs)
		done := pairs * 2
		if done < blocks {
			packSSEtab[bits](dst[done*wb:], src[done*BlockSize:], 1)
		}
		return
	}
	packSSEtab[bits](dst, src, blocks)
}

// unpackBulk is the inverse of packBulk.
func unpackBulk(dst []uint32, src []byte, bits, blocks int) {
	wb := 16 * bits
	if hasAVX2 && blocks >= 2 {
		pairs := blocks / 2
		unpackAVX2tab[bits](dst, src, pairs)
		done := pairs * 2
		if done < blocks {
			unpackSSEtab[bits](dst[done*BlockSize:], src[done*wb:], 1)
		}
		return
	}
	unpackSSEtab[bits](dst, src, blocks)
}

// packBlock/unpackBlock satisfy the single-block contract (used by tests).
func packBlock(dst []byte, src []uint32, bits int)   { packSSEtab[bits](dst, src, 1) }
func unpackBlock(dst []uint32, src []byte, bits int) { unpackSSEtab[bits](dst, src, 1) }
